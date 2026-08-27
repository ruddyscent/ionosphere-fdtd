"""Vectorized 3-D geodesic FDTD time stepping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._torch_runtime import _TorchRuntime
from .constants import C_0, EARTH_RADIUS_M, EPSILON_0, MU_0
from .materials import (
    EarthIonosphereMaterial,
    MaterialUpdateCoefficientTensors,
    SampledMaterialTensors,
    apply_fractional_cell_anomalies,
    apply_fractional_point_anomalies,
    conservative_anomaly_fractions,
)
from .mesh import GeodesicMesh, build_geodesic_mesh
from .plasma import (
    GeodesicPlasmaCoupler,
    MeshPlasmaModel,
    PlasmaCoefficientTensors,
    PlasmaSpeciesCoefficientTensors,
)
from .radial_grid import validate_radial_grid
from .sources import GaussianCurrent, TangentialGaussianCurrent
from .surface_impedance import (
    ConductiveHalfSpaceSurface,
    SurfaceImpedanceADE,
    SurfaceImpedanceCoefficientTensors,
)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Geometry and integration controls for a simulation."""

    subdivision: int = 2
    radial_cells: int = 24
    minimum_altitude_m: float = -100_000.0
    maximum_altitude_m: float = 100_000.0
    earth_radius_m: float = EARTH_RADIUS_M
    courant_factor: float = 0.35
    time_step_s: float | None = None
    mesh_relaxations: int = 0
    mesh_optimization_steps: int = 0
    mesh_orientation: str = "polar"
    radial_altitudes_m: tuple[float, ...] | None = None
    radial_material_support: str = "point"
    tangential_material_support: str = "point"
    horizontal_anomaly_mode: str = "point"
    radial_boundary_condition: str = "pec"
    loss_integration: str = "exponential"
    radial_grid_policy: str = "smooth"
    geometry_mode: str = "full-spherical"
    compress_uniform_material_coefficients: bool = False

    def __post_init__(self) -> None:
        integer_controls = {
            "subdivision": self.subdivision,
            "radial_cells": self.radial_cells,
            "mesh_relaxations": self.mesh_relaxations,
            "mesh_optimization_steps": self.mesh_optimization_steps,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in integer_controls.values()
        ):
            raise ValueError("mesh and radial cell controls must be integers")
        if self.subdivision < 0:
            raise ValueError("subdivision must be non-negative")
        if self.radial_cells < 2:
            raise ValueError("radial_cells must be at least 2")
        if self.mesh_relaxations < 0:
            raise ValueError("mesh_relaxations must be non-negative")
        if self.mesh_optimization_steps < 0:
            raise ValueError("mesh_optimization_steps must be non-negative")
        if self.mesh_orientation not in {"native", "polar"}:
            raise ValueError("mesh_orientation must be 'native' or 'polar'")
        if self.radial_material_support not in {"point", "dual-cell"}:
            raise ValueError(
                "radial_material_support must be 'point' or 'dual-cell'"
            )
        if self.tangential_material_support not in {"point", "edge-diamond"}:
            raise ValueError(
                "tangential_material_support must be 'point' or 'edge-diamond'"
            )
        if self.horizontal_anomaly_mode not in {"point", "conservative-nearest"}:
            raise ValueError(
                "horizontal_anomaly_mode must be 'point' or 'conservative-nearest'"
            )
        if self.radial_boundary_condition not in {"pec", "surface-impedance"}:
            raise ValueError(
                "radial_boundary_condition must be 'pec' or 'surface-impedance'"
            )
        if (
            self.radial_boundary_condition == "surface-impedance"
            and self.minimum_altitude_m != 0.0
        ):
            raise ValueError(
                "surface-impedance boundary requires minimum_altitude_m=0"
            )
        if self.loss_integration not in {"exponential", "trapezoidal"}:
            raise ValueError(
                "loss_integration must be 'exponential' or 'trapezoidal'"
            )
        if self.radial_grid_policy not in {
            "smooth",
            "balanced-2to1",
            "allow-abrupt",
        }:
            raise ValueError(
                "radial_grid_policy must be 'smooth', 'balanced-2to1', "
                "or 'allow-abrupt'"
            )
        if self.geometry_mode not in {"full-spherical", "thin-shell"}:
            raise ValueError(
                "geometry_mode must be 'full-spherical' or 'thin-shell'"
            )
        finite_geometry = (
            self.minimum_altitude_m,
            self.maximum_altitude_m,
            self.earth_radius_m,
            self.courant_factor,
        )
        if not all(np.isfinite(value) for value in finite_geometry):
            raise ValueError("geometry and Courant controls must be finite")
        if self.minimum_altitude_m >= self.maximum_altitude_m:
            raise ValueError("altitude bounds are reversed")
        if self.earth_radius_m + self.minimum_altitude_m <= 0.0:
            raise ValueError("minimum radius must be positive")
        if not 0.0 < self.courant_factor <= 1.0:
            raise ValueError("courant_factor must be in (0, 1]")
        if self.time_step_s is not None and (
            not np.isfinite(self.time_step_s) or self.time_step_s <= 0.0
        ):
            raise ValueError("time_step_s must be finite and positive")
        if self.radial_altitudes_m is not None:
            altitudes = np.asarray(self.radial_altitudes_m, dtype=np.float64)
            if (
                altitudes.ndim != 1
                or len(altitudes) < 3
                or not np.all(np.isfinite(altitudes))
                or not np.all(np.diff(altitudes) > 0.0)
            ):
                raise ValueError(
                    "radial_altitudes_m must contain at least three finite, "
                    "increasing values"
                )
            if len(altitudes) != self.radial_cells + 1:
                raise ValueError(
                    "radial_cells must equal len(radial_altitudes_m) - 1"
                )
            if (
                altitudes[0] != self.minimum_altitude_m
                or altitudes[-1] != self.maximum_altitude_m
            ):
                raise ValueError(
                    "custom radial altitudes must include the configured altitude bounds"
                )
            radial_steps = np.diff(altitudes)
            normalized_step_change = self.radial_cells * np.abs(
                np.diff(radial_steps)
            ) / np.minimum(radial_steps[:-1], radial_steps[1:])
            if self.radial_grid_policy == "smooth" and np.any(
                normalized_step_change > 8.0 * (1.0 + 1.0e-12)
            ):
                raise ValueError(
                    "custom radial grid is not smoothly graded; use a smooth node "
                    "mapping or explicitly select radial_grid_policy='allow-abrupt'"
                )
            if self.radial_grid_policy == "balanced-2to1":
                validate_radial_grid(
                    altitudes, maximum_adjacent_step_ratio=2.0
                )


@dataclass(frozen=True, slots=True)
class _StaticSimulationData:
    """Host-prepared geometry, material, source, and time-step data."""

    altitudes_m: NDArray[np.float64]
    radii_m: NDArray[np.float64]
    radial_midpoints_m: NDArray[np.float64]
    radial_midpoint_altitudes_m: NDArray[np.float64]
    radial_steps_m: NDArray[np.float64]
    radial_node_control_lengths_m: NDArray[np.float64]
    sigma_er: NDArray[np.float64]
    epsilon_r_er: NDArray[np.float64]
    sigma_et: NDArray[np.float64]
    epsilon_r_et: NDArray[np.float64]
    cfl_time_step_limit_s: float
    maximum_stable_time_step_s: float
    time_step_s: float
    source_distribution: tuple[
        NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]
    ] | None
    tangential_source_distribution: tuple[
        NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]
    ] | None
    anomaly_horizontal_fractions_er: tuple[NDArray[np.float64], ...] | None
    anomaly_horizontal_fractions_et: tuple[NDArray[np.float64], ...] | None


class GeodesicFDTD:
    """Earth-ionosphere FDTD model using staggered geodesic radial planes.

    ``er`` and ``ht`` live on integer radial planes (TM-r), while ``hr`` and
    ``et`` live halfway between them (TE-r).  Magnetic fields are staggered by
    half a time step from electric fields, as in the Yee algorithm. The default
    radial curls retain the full spherical metric; paper reproduction helpers
    explicitly select the legacy thin-shell approximation.
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        material: EarthIonosphereMaterial | None = None,
        source: GaussianCurrent | TangentialGaussianCurrent | None = None,
        surface_impedance: ConductiveHalfSpaceSurface | None = None,
        plasma: MeshPlasmaModel | None = None,
        mesh: GeodesicMesh | None = None,
        device: str = "cpu",
        dtype: str = "float64",
        compile_step: bool = False,
        compile_chunk_size: int = 8,
        torch_threads: int | None = None,
        material_tensors: (
            SampledMaterialTensors | MaterialUpdateCoefficientTensors | None
        ) = None,
        surface_impedance_tensors: (
            SurfaceImpedanceCoefficientTensors | None
        ) = None,
        plasma_tensors: PlasmaCoefficientTensors | None = None,
    ) -> None:
        if (
            isinstance(compile_chunk_size, bool)
            or not isinstance(compile_chunk_size, (int, np.integer))
            or compile_chunk_size < 1
        ):
            raise ValueError("compile_chunk_size must be a positive integer")
        self.compile_chunk_size = int(compile_chunk_size)
        self.config = config or SimulationConfig()
        if mesh is None:
            self.mesh = build_geodesic_mesh(
                subdivision=self.config.subdivision,
                relaxations=self.config.mesh_relaxations,
                orientation=self.config.mesh_orientation,
                optimization_steps=self.config.mesh_optimization_steps,
            )
        else:
            if self.config.mesh_relaxations or self.config.mesh_optimization_steps:
                raise ValueError(
                    "mesh relaxation and optimization controls cannot accompany "
                    "a provided mesh"
                )
            if mesh.topology_kind == "uniform":
                polar_pentagons = np.count_nonzero(
                    (mesh.vertex_degree == 5)
                    & np.isclose(
                        np.abs(mesh.vertices[:, 2]), 1.0, rtol=0.0, atol=1.0e-13
                    )
                )
                expected_polar_pentagons = (
                    2 if self.config.mesh_orientation == "polar" else 0
                )
                if polar_pentagons != expected_polar_pentagons:
                    raise ValueError("provided mesh orientation does not match config")
            self.mesh = mesh
        if (
            self.mesh.subdivision is not None
            and self.mesh.subdivision != self.config.subdivision
        ):
            raise ValueError("provided mesh subdivision does not match config")
        self._runtime = _TorchRuntime(
            self.mesh,
            device=device,
            dtype=dtype,
            threads=torch_threads,
        )
        self.material = material or EarthIonosphereMaterial()
        self.source = source
        self.surface_impedance = surface_impedance
        self.material_tensors = material_tensors
        self.plasma = plasma
        self.surface_impedance_tensors = surface_impedance_tensors
        self.plasma_tensors = plasma_tensors
        if self.config.radial_boundary_condition == "surface-impedance":
            if surface_impedance is None:
                raise ValueError(
                    "surface-impedance boundary requires a surface model"
                )
        elif surface_impedance is not None:
            raise ValueError(
                "surface impedance model requires radial_boundary_condition="
                "'surface-impedance'"
            )

        static_data = _prepare_static_simulation(
            self.config,
            self.mesh,
            self.material,
            self.source,
            self.plasma,
        )
        for name in (
            "altitudes_m",
            "radii_m",
            "radial_midpoints_m",
            "radial_midpoint_altitudes_m",
            "radial_steps_m",
            "radial_node_control_lengths_m",
            "sigma_er",
            "epsilon_r_er",
            "sigma_et",
            "epsilon_r_et",
            "cfl_time_step_limit_s",
            "maximum_stable_time_step_s",
            "time_step_s",
        ):
            setattr(self, name, getattr(static_data, name))
        if static_data.anomaly_horizontal_fractions_er is not None:
            self.anomaly_horizontal_fractions_er = (
                static_data.anomaly_horizontal_fractions_er
            )
        if static_data.anomaly_horizontal_fractions_et is not None:
            self.anomaly_horizontal_fractions_et = (
                static_data.anomaly_horizontal_fractions_et
            )

        self.er = self._runtime.zeros((self.mesh.n_vertices, len(self.radii_m)))
        self.ht = self._runtime.zeros((self.mesh.n_edges, len(self.radii_m)))
        self.et = self._runtime.zeros(
            (self.mesh.n_edges, len(self.radial_midpoints_m))
        )
        self.hr = self._runtime.zeros(
            (self.mesh.n_faces, len(self.radial_midpoints_m))
        )
        self._surface_impedance_ade = (
            SurfaceImpedanceADE(
                surface_impedance,
                edge_count=self.mesh.n_edges,
                time_step_s=self.time_step_s,
                runtime=self._runtime,
            )
            if surface_impedance is not None
            else None
        )
        self._plasma_coupler = (
            GeodesicPlasmaCoupler(
                plasma,
                self.mesh,
                self.radial_midpoint_altitudes_m,
                self.time_step_s,
                self._runtime,
            )
            if plasma is not None
            else None
        )

        self._prepare_geometry()
        self._prepare_optional_physics_coefficients()
        self._prepare_material_coefficients(static_data)
        self._source_distribution = None
        self._tangential_source_distribution = None
        if static_data.tangential_source_distribution is not None:
            edges, layers, weights = static_data.tangential_source_distribution
            self._tangential_source_distribution = (
                self._runtime.index_tensor(edges),
                self._runtime.index_tensor(layers),
                self._runtime.as_tensor(weights),
            )
        elif static_data.source_distribution is not None:
            vertices, layers, weights = static_data.source_distribution
            self._source_distribution = (
                self._runtime.index_tensor(vertices),
                self._runtime.index_tensor(layers),
                self._runtime.as_tensor(weights),
            )
        self.time_s = 0.0
        self.steps = 0
        self.compiled = compile_step
        self._initialize_field_stepper(compile_step)

    def _prepare_optional_physics_coefficients(self) -> None:
        """Validate tensor-native ADE coefficients and preserve their graphs."""

        if self.surface_impedance_tensors is not None:
            if self._surface_impedance_ade is None:
                raise ValueError(
                    "surface_impedance_tensors requires a surface impedance model"
                )
            tensors = self.surface_impedance_tensors
            if not isinstance(tensors, SurfaceImpedanceCoefficientTensors):
                raise TypeError(
                    "surface_impedance_tensors must be "
                    "SurfaceImpedanceCoefficientTensors"
                )
            terms_shape = (self.surface_impedance.terms,)
            scale_shape = (self.mesh.n_edges,)
            decay = self._validated_physics_tensor(
                tensors.decay,
                terms_shape,
                "surface_impedance_tensors.decay",
                minimum=-1.0,
                maximum=1.0,
            )
            drive = self._validated_physics_tensor(
                tensors.drive,
                terms_shape,
                "surface_impedance_tensors.drive",
                minimum=0.0,
                maximum=1.0,
            )
            history_weights = self._validated_physics_tensor(
                tensors.history_weights,
                terms_shape,
                "surface_impedance_tensors.history_weights",
                minimum=0.0,
                strict_minimum=True,
            )
            scale = self._validated_physics_tensor(
                tensors.scale,
                scale_shape,
                "surface_impedance_tensors.scale",
                minimum=0.0,
                strict_minimum=True,
            )
            self.surface_impedance_tensors = (
                SurfaceImpedanceCoefficientTensors(
                    decay, drive, history_weights, scale
                )
            )
            ade = self._surface_impedance_ade
            ade._decay = decay
            ade._drive = drive
            ade._history_weights = history_weights
            ade._scale = scale

        if self.plasma_tensors is not None:
            if self._plasma_coupler is None:
                raise ValueError("plasma_tensors requires a plasma model")
            tensors = self.plasma_tensors
            if not isinstance(tensors, PlasmaCoefficientTensors):
                raise TypeError(
                    "plasma_tensors must be PlasmaCoefficientTensors"
                )
            if not isinstance(tensors.species, tuple):
                raise TypeError("plasma_tensors.species must be a tuple")
            ade = self._plasma_coupler.ade
            if len(tensors.species) != len(ade._coefficients):
                raise ValueError(
                    "plasma_tensors.species must match the plasma model species"
                )
            vector_shape = tuple(ade._magnetic_direction.shape)
            coefficient_shape = vector_shape[:2]
            magnetic_direction = self._validated_physics_tensor(
                tensors.magnetic_direction,
                vector_shape,
                "plasma_tensors.magnetic_direction",
            )
            normalized_species = []
            normalized_coefficients = []
            rules = (
                ("decay", 0.0, False, 1.0),
                ("cosine", -1.0, False, 1.0),
                ("sine", -1.0, False, 1.0),
                ("drive_parallel", 0.0, False, None),
                ("drive_real", None, False, None),
                ("drive_imag", None, False, None),
            )
            for index, species in enumerate(tensors.species):
                if not isinstance(species, PlasmaSpeciesCoefficientTensors):
                    raise TypeError(
                        f"plasma_tensors.species[{index}] must be "
                        "PlasmaSpeciesCoefficientTensors"
                    )
                values = []
                for name, minimum, strict_minimum, maximum in rules:
                    values.append(
                        self._validated_physics_tensor(
                            getattr(species, name),
                            coefficient_shape,
                            f"plasma_tensors.species[{index}].{name}",
                            minimum=minimum,
                            strict_minimum=strict_minimum,
                            maximum=maximum,
                        )
                    )
                normalized_species.append(
                    PlasmaSpeciesCoefficientTensors(*values)
                )
                normalized_coefficients.append(tuple(values))
            self.plasma_tensors = PlasmaCoefficientTensors(
                magnetic_direction, tuple(normalized_species)
            )
            ade._magnetic_direction = magnetic_direction
            ade._coefficients = normalized_coefficients

    def _initialize_field_stepper(self, compile_step: bool) -> None:
        """Configure the existing NumPy or functional PyTorch recurrence."""

        from ._torch_step import (
            FieldState,
            FieldStepParameters,
            OptionalPhysicsState,
            OptionalPhysicsStepParameters,
            PlasmaSpeciesStepParameters,
            PlasmaStepParameters,
            SurfaceImpedanceStepParameters,
            _optional_physics_requires_grad,
            advance,
            advance_chunk,
            advance_electric,
            advance_magnetic,
            advance_optional_physics,
            advance_optional_physics_chunk,
            lower_surface_gradient_er,
        )

        self._field_state_type = FieldState
        self._advance_electric_state = advance_electric
        self._advance_magnetic_state = advance_magnetic
        self._optional_physics_state_type = OptionalPhysicsState
        self._optional_physics_requires_grad = _optional_physics_requires_grad
        self._lower_surface_gradient_er = lower_surface_gradient_er
        radial_vertices = radial_layers = radial_weights = None
        if self._source_distribution is not None:
            radial_vertices, radial_layers, radial_weights = (
                self._source_distribution
            )
        tangential_edges = tangential_layers = tangential_weights = None
        if self._tangential_source_distribution is not None:
            tangential_edges, tangential_layers, tangential_weights = (
                self._tangential_source_distribution
            )
        self._field_step_parameters = FieldStepParameters(
            time_step_over_mu=self.time_step_s / MU_0,
            full_spherical_geometry=(
                self.config.geometry_mode == "full-spherical"
            ),
            compress_uniform_material_coefficients=(
                self.config.compress_uniform_material_coefficients
            ),
            edges=self._runtime.edges,
            face_edges=self._runtime.face_edges,
            face_edge_signs=self._runtime.face_edge_signs,
            edge_left_faces=self._runtime.edge_left_faces,
            edge_right_faces=self._runtime.edge_right_faces,
            vertex_edges=self._runtime.vertex_edges,
            vertex_edge_signs=self._runtime.vertex_edge_signs,
            primal_edge_angles=self._primal_edge_angles,
            inverse_primal_edge_angles=self._inverse_primal_edge_angles,
            dual_edge_angles=self._dual_edge_angles,
            inverse_dual_edge_angles=self._inverse_dual_edge_angles,
            inverse_dual_cell_solid_angles=(
                self._inverse_dual_cell_solid_angles
            ),
            inverse_face_solid_angles=self._inverse_face_solid_angles,
            radii=self._radii,
            inverse_radii=self._inverse_radii,
            radial_midpoints=self._radial_midpoints,
            inverse_radial_midpoints=self._inverse_radial_midpoints,
            radial_steps=self._radial_steps,
            radial_node_control_lengths=self._radial_node_control_lengths,
            radial_center_distances=self._radial_center_distances,
            ca_er=self._ca_er,
            cb_er=self._cb_er,
            ca_et=self._ca_et,
            cb_et=self._cb_et,
            radial_source_vertices=radial_vertices,
            radial_source_layers=radial_layers,
            radial_source_weights=radial_weights,
            radial_source_element_length=(
                self.source.vertical_element_length_m
                if self._source_distribution is not None
                else 0.0
            ),
            tangential_source_edges=tangential_edges,
            tangential_source_layers=tangential_layers,
            tangential_source_weights=tangential_weights,
        )
        surface_parameters = None
        if self._surface_impedance_ade is not None:
            boundary_metric = (
                self.radial_midpoints_m[0] / self.radii_m[0]
                if self.config.geometry_mode == "full-spherical"
                else 1.0
            )
            ade = self._surface_impedance_ade
            surface_parameters = SurfaceImpedanceStepParameters(
                ade._decay,
                ade._drive,
                ade._history_weights,
                ade._scale,
                boundary_metric,
                self._radial_steps[0],
            )
        plasma_parameters = None
        if self._plasma_coupler is not None:
            coupler = self._plasma_coupler
            plasma_parameters = PlasmaStepParameters(
                coupler.ade._magnetic_direction,
                tuple(
                    PlasmaSpeciesStepParameters(*coefficients)
                    for coefficients in coupler.ade._coefficients
                ),
                coupler._face_edges,
                coupler._faces,
                coupler._reconstruction,
                coupler._face_centers,
                coupler._left_faces,
                coupler._right_faces,
                coupler._left_tangents,
                coupler._right_tangents,
                coupler._vertex_faces,
                coupler._vertex_face_weights,
            )
        self._optional_physics_step_parameters = OptionalPhysicsStepParameters(
            self._field_step_parameters,
            surface_parameters,
            plasma_parameters,
        )
        self._has_optional_physics = (
            surface_parameters is not None or plasma_parameters is not None
        )

        runtime = self._runtime
        if self._has_optional_physics:
            self._tensor_optional_physics_step = (
                runtime.compile(advance_optional_physics)
                if compile_step
                else advance_optional_physics
            )
            self._tensor_optional_physics_chunk = (
                runtime.compile(advance_optional_physics_chunk)
                if compile_step
                else advance_optional_physics_chunk
            )
            if compile_step and runtime.device.type == "cpu":
                self._tensor_optional_physics_gradient_step = runtime.compile(
                    advance_optional_physics, backend="aot_eager"
                )
                self._tensor_optional_physics_gradient_chunk = runtime.compile(
                    advance_optional_physics_chunk, backend="aot_eager"
                )
            else:
                self._tensor_optional_physics_gradient_step = (
                    self._tensor_optional_physics_step
                )
                self._tensor_optional_physics_gradient_chunk = (
                    self._tensor_optional_physics_chunk
                )
        else:
            self._tensor_field_step = (
                runtime.compile(advance) if compile_step else advance
            )
            self._tensor_field_chunk = (
                runtime.compile(advance_chunk) if compile_step else advance_chunk
            )
        self._field_step = self._advance_torch_fields
        self._field_chunk = self._advance_torch_field_chunk

    def _estimate_cfl_time_step_limit(self) -> float:
        """Return the conservative lossless CFL limit before the safety factor."""

        smallest_radius = float(self.radii_m.min())
        primal = smallest_radius * float(self.mesh.primal_edge_angles.min())
        dual = smallest_radius * float(self.mesh.dual_edge_angles.min())
        radial = float(self.radial_steps_m.min())
        inverse_length_squared = primal**-2 + dual**-2 + (2.0 / radial) ** 2
        minimum_epsilon_r = min(
            float(np.min(self.epsilon_r_er)),
            float(np.min(self.epsilon_r_et)),
        )
        geometric_limit = np.sqrt(minimum_epsilon_r) / (
            C_0 * np.sqrt(inverse_length_squared)
        )
        if self.plasma is None:
            return geometric_limit
        plasma_drive = np.zeros(self.plasma.magnetic_field_t.shape[:2])
        for species in self.plasma.species:
            plasma_drive += (
                species.number_density_m3
                * species.charge_c**2
                / species.mass_kg
            )
        maximum_plasma_frequency = np.sqrt(
            float(np.max(plasma_drive)) / (EPSILON_0 * minimum_epsilon_r)
        )
        if maximum_plasma_frequency == 0.0:
            return geometric_limit
        return min(geometric_limit, 2.0 / maximum_plasma_frequency)

    def _prepare_geometry(self) -> None:
        # Spherical metric tensors are separable into horizontal angles and
        # radial factors. Keeping those factors one-dimensional avoids several
        # dense edge-by-layer and face-by-layer arrays on the accelerator.
        self._primal_edge_angles = self._runtime.as_tensor(
            self.mesh.primal_edge_angles[:, None]
        )
        self._inverse_primal_edge_angles = self._runtime.as_tensor(
            1.0 / self.mesh.primal_edge_angles[:, None]
        )
        self._dual_edge_angles = self._runtime.as_tensor(
            self.mesh.dual_edge_angles[:, None]
        )
        self._inverse_dual_edge_angles = self._runtime.as_tensor(
            1.0 / self.mesh.dual_edge_angles[:, None]
        )
        self._inverse_dual_cell_solid_angles = self._runtime.as_tensor(
            1.0 / self.mesh.dual_cell_solid_angles[:, None]
        )
        self._inverse_face_solid_angles = self._runtime.as_tensor(
            1.0 / self.mesh.face_solid_angles[:, None]
        )
        self._radii = self._runtime.as_tensor(self.radii_m)
        self._inverse_radii = self._runtime.as_tensor(1.0 / self.radii_m[None, :])
        self._radial_midpoints = self._runtime.as_tensor(self.radial_midpoints_m)
        self._inverse_radial_midpoints = self._runtime.as_tensor(
            1.0 / self.radial_midpoints_m[None, :]
        )
        self._radial_steps = self._runtime.as_tensor(self.radial_steps_m)
        self._radial_node_control_lengths = self._runtime.as_tensor(
            self.radial_node_control_lengths_m
        )
        self._radial_center_distances = self._runtime.as_tensor(
            self.radial_midpoints_m[1:] - self.radial_midpoints_m[:-1]
        )
    def _radial_derivative_et(self) -> Any:
        values = self.et
        if self.config.geometry_mode == "full-spherical":
            values = values * self._radial_midpoints[None, :]
        result = self._runtime.empty_like(self.ht)
        result[:, 0] = 2.0 * values[:, 0] / self._radial_steps[0]
        result[:, -1] = -2.0 * values[:, -1] / self._radial_steps[-1]
        if self.ht.shape[1] > 2:
            result[:, 1:-1] = self._runtime.diff(
                values, axis=1
            ) / self._radial_center_distances
        if self.config.geometry_mode == "full-spherical":
            result *= self._inverse_radii
        return result

    def _radial_derivative_ht(self) -> Any:
        values = self.ht
        if self.config.geometry_mode == "full-spherical":
            values = values * self._radii[None, :]
        result = self._runtime.diff(values, axis=1) / self._radial_steps[None, :]
        if self.config.geometry_mode == "full-spherical":
            result *= self._inverse_radial_midpoints
        return result

    def _sample_material_properties(self) -> None:
        """Sample validated host-side material properties before choosing dt."""

        mesh_sampler = getattr(self.material, "sample_mesh", None)
        if mesh_sampler is not None:
            values = mesh_sampler(
                self.mesh,
                self.altitudes_m,
                self.config.earth_radius_m,
                radial_material_support=self.config.radial_material_support,
                tangential_material_support=(
                    self.config.tangential_material_support
                ),
                horizontal_anomaly_mode=self.config.horizontal_anomaly_mode,
            )
            try:
                sigma_er, epsilon_r_er, sigma_et, epsilon_r_et = values
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "mesh material sampler must return four property arrays"
                ) from error
            self.sigma_er, self.epsilon_r_er = self._validated_material_sample(
                (sigma_er, epsilon_r_er),
                (self.mesh.n_vertices, len(self.altitudes_m)),
                "mesh-native radial",
            )
            self.sigma_et, self.epsilon_r_et = self._validated_material_sample(
                (sigma_et, epsilon_r_et),
                (self.mesh.n_edges, len(self.radial_midpoint_altitudes_m)),
                "mesh-native tangential",
            )
            return

        material_anomalies = getattr(self.material, "anomalies", None)
        if (
            self.config.horizontal_anomaly_mode == "conservative-nearest"
            and material_anomalies is None
        ):
            raise ValueError(
                "conservative-nearest anomalies require a material with an "
                "anomalies collection"
            )
        conservative_anomalies = (
            self.config.horizontal_anomaly_mode == "conservative-nearest"
            and bool(material_anomalies)
        )
        if conservative_anomalies:
            try:
                sampling_material = replace(self.material, anomalies=())
            except TypeError as error:
                raise ValueError(
                    "conservative-nearest anomalies require a dataclass material"
                ) from error
        else:
            sampling_material = self.material
        if self.config.radial_material_support == "point":
            sigma_er, epsilon_r_er = self._validated_material_sample(
                sampling_material.sample(
                    self.mesh.vertices,
                    self.altitudes_m,
                    self.config.earth_radius_m,
                ),
                (self.mesh.n_vertices, len(self.altitudes_m)),
                "radial",
            )
        else:
            sigma_er, epsilon_r_er = self._dual_cell_material_average(
                sampling_material
            )
        if conservative_anomalies:
            anomalies = tuple(material_anomalies)
            fractions_er = tuple(
                conservative_anomaly_fractions(
                    self.mesh.vertices,
                    self.mesh.dual_cell_solid_angles,
                    anomaly,
                    self.config.earth_radius_m,
                )
                for anomaly in anomalies
            )
            apply_fractional_point_anomalies(
                sigma_er,
                epsilon_r_er,
                self.altitudes_m,
                anomalies,
                fractions_er,
            )
            self.anomaly_horizontal_fractions_er = fractions_er

        def sample_tangential(
            directions: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            sample_cells = getattr(sampling_material, "sample_tangential_cells", None)
            if sample_cells is None:
                values = sampling_material.sample(
                    directions,
                    self.radial_midpoint_altitudes_m,
                    self.config.earth_radius_m,
                )
            else:
                values = sample_cells(
                    directions,
                    self.altitudes_m[:-1],
                    self.altitudes_m[1:],
                    self.config.earth_radius_m,
                )
            return self._validated_material_sample(
                values,
                (len(directions), len(self.radial_midpoint_altitudes_m)),
                "tangential",
            )

        edge_midpoints = self.mesh.edge_midpoints()
        if self.config.tangential_material_support == "point":
            sigma_et, epsilon_r_et = sample_tangential(edge_midpoints)
        else:
            endpoints = self.mesh.vertices[self.mesh.edges]
            left = self.mesh.face_centers[self.mesh.edge_left_faces]
            right = self.mesh.face_centers[self.mesh.edge_right_faces]
            support_directions = (
                edge_midpoints + endpoints[:, 0] + left,
                edge_midpoints + left + endpoints[:, 1],
                edge_midpoints + endpoints[:, 1] + right,
                edge_midpoints + right + endpoints[:, 0],
            )
            sigma_et = np.zeros(
                (self.mesh.n_edges, len(self.radial_midpoints_m)),
                dtype=np.float64,
            )
            epsilon_r_et = np.zeros_like(sigma_et)
            quadrant_areas = self.mesh.edge_diamond_quadrant_solid_angles()
            quadrant_weights = quadrant_areas / np.sum(
                quadrant_areas, axis=1, keepdims=True
            )
            for quadrant, directions in enumerate(support_directions):
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                support_sigma, support_epsilon = sample_tangential(directions)
                weight = quadrant_weights[:, quadrant, None]
                sigma_et += weight * support_sigma
                epsilon_r_et += weight * support_epsilon
        if conservative_anomalies:
            fractions_et = tuple(
                conservative_anomaly_fractions(
                    edge_midpoints,
                    self.mesh.edge_diamond_solid_angles(),
                    anomaly,
                    self.config.earth_radius_m,
                )
                for anomaly in anomalies
            )
            apply_fractional_cell_anomalies(
                sigma_et,
                epsilon_r_et,
                self.altitudes_m[:-1],
                self.altitudes_m[1:],
                anomalies,
                fractions_et,
            )
            self.anomaly_horizontal_fractions_et = fractions_et
        self.sigma_er = sigma_er
        self.sigma_et = sigma_et
        self.epsilon_r_er = epsilon_r_er
        self.epsilon_r_et = epsilon_r_et

    def _dual_cell_material_average(
        self, sampling_material: Any
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Area-average radial-field material over each polygonal dual cell."""

        vertex_count = self.mesh.n_vertices
        layer_count = len(self.altitudes_m)
        maximum_degree = int(np.max(self.mesh.vertex_degree))
        incidence = np.full(
            (vertex_count, maximum_degree), -1, dtype=np.int64
        )
        incidence_vertices = self.mesh.edges.ravel()
        incidence_edges = np.repeat(
            np.arange(self.mesh.n_edges, dtype=np.int64), 2
        )
        order = np.argsort(incidence_vertices, kind="stable")
        sorted_vertices = incidence_vertices[order]
        counts = self.mesh.vertex_degree
        starts = np.repeat(
            np.cumsum(np.r_[0, counts[:-1]], dtype=np.int64), counts
        )
        slots = np.arange(len(order), dtype=np.int64) - starts
        incidence[sorted_vertices, slots] = incidence_edges[order]

        sigma = np.empty((vertex_count, layer_count), dtype=np.float64)
        epsilon_r = np.empty_like(sigma)
        target_support_count = 65_536
        vertices_per_chunk = max(1, target_support_count // maximum_degree)
        for begin in range(0, vertex_count, vertices_per_chunk):
            end = min(begin + vertices_per_chunk, vertex_count)
            chunk_incidence = incidence[begin:end]
            valid = chunk_incidence >= 0
            local_vertices = np.broadcast_to(
                np.arange(end - begin, dtype=np.int64)[:, None],
                chunk_incidence.shape,
            )[valid]
            global_vertices = begin + local_vertices
            edge_indices = chunk_incidence[valid]
            directions, wedge_areas = self.mesh.dual_cell_wedge_quadrature(
                global_vertices, edge_indices
            )
            weights = (
                wedge_areas
                / self.mesh.dual_cell_solid_angles[global_vertices]
            )
            support_sigma, support_epsilon = self._validated_material_sample(
                sampling_material.sample(
                    directions,
                    self.altitudes_m,
                    self.config.earth_radius_m,
                ),
                (len(directions), layer_count),
                "radial dual-cell support",
            )
            chunk_sigma = np.zeros((end - begin, layer_count), dtype=np.float64)
            chunk_epsilon = np.zeros_like(chunk_sigma)
            np.add.at(
                chunk_sigma,
                local_vertices,
                weights[:, None] * support_sigma,
            )
            np.add.at(
                chunk_epsilon,
                local_vertices,
                weights[:, None] * support_epsilon,
            )
            weight_sums = np.bincount(
                local_vertices, weights=weights, minlength=end - begin
            )
            if not np.allclose(weight_sums, 1.0, rtol=0.0, atol=2.0e-12):
                raise RuntimeError("dual-cell material weights do not close")
            sigma[begin:end] = chunk_sigma
            epsilon_r[begin:end] = chunk_epsilon
        return sigma, epsilon_r

    def _prepare_material_coefficients(
        self, static_data: _StaticSimulationData
    ) -> None:
        """Build lossy electric-field update coefficients at the selected dt."""

        if self.material_tensors is not None:
            self._prepare_tensor_material_coefficients()
            return

        ca_er, cb_er, ca_et, cb_et = _host_material_update_coefficients(
            static_data, self.config
        )
        self._ca_er = self._runtime.as_tensor(ca_er)
        self._cb_er = self._runtime.as_tensor(cb_er)
        self._ca_et = self._runtime.as_tensor(ca_et)
        self._cb_et = self._runtime.as_tensor(cb_et)

    def _prepare_tensor_material_coefficients(self) -> None:
        """Validate tensor-native material inputs and retain their graph."""

        er_shape, et_shape = self._material_tensor_shapes()
        tensors = self.material_tensors
        if isinstance(tensors, SampledMaterialTensors):
            sigma_er = self._validated_material_tensor(
                tensors.sigma_er, er_shape, "sigma_er", minimum=0.0
            )
            epsilon_r_er = self._validated_material_tensor(
                tensors.epsilon_r_er,
                er_shape,
                "epsilon_r_er",
                minimum=0.0,
                strict_minimum=True,
            )
            sigma_et = self._validated_material_tensor(
                tensors.sigma_et, et_shape, "sigma_et", minimum=0.0
            )
            epsilon_r_et = self._validated_material_tensor(
                tensors.epsilon_r_et,
                et_shape,
                "epsilon_r_et",
                minimum=0.0,
                strict_minimum=True,
            )
            self.material_tensors = SampledMaterialTensors(
                sigma_er, epsilon_r_er, sigma_et, epsilon_r_et
            )
            epsilon_er = EPSILON_0 * epsilon_r_er
            epsilon_et = EPSILON_0 * epsilon_r_et
            if self.config.loss_integration == "trapezoidal":
                loss_er = sigma_er * self.time_step_s / (2.0 * epsilon_er)
                loss_et = sigma_et * self.time_step_s / (2.0 * epsilon_et)
                self._ca_er = (1.0 - loss_er) / (1.0 + loss_er)
                self._cb_er = self.time_step_s / (
                    epsilon_er * (1.0 + loss_er)
                )
                self._ca_et = (1.0 - loss_et) / (1.0 + loss_et)
                self._cb_et = self.time_step_s / (
                    epsilon_et * (1.0 + loss_et)
                )
            else:
                self._ca_er, self._cb_er = (
                    self._torch_exponential_loss_coefficients(
                        sigma_er, epsilon_er
                    )
                )
                self._ca_et, self._cb_et = (
                    self._torch_exponential_loss_coefficients(
                        sigma_et, epsilon_et
                    )
                )
            return
        if isinstance(tensors, MaterialUpdateCoefficientTensors):
            ca_er = self._validated_material_tensor(
                tensors.ca_er,
                er_shape,
                "ca_er",
                minimum=-1.0,
                strict_minimum=True,
                maximum=1.0,
            )
            cb_er = self._validated_material_tensor(
                tensors.cb_er,
                er_shape,
                "cb_er",
                minimum=0.0,
                strict_minimum=True,
            )
            ca_et = self._validated_material_tensor(
                tensors.ca_et,
                et_shape,
                "ca_et",
                minimum=-1.0,
                strict_minimum=True,
                maximum=1.0,
            )
            cb_et = self._validated_material_tensor(
                tensors.cb_et,
                et_shape,
                "cb_et",
                minimum=0.0,
                strict_minimum=True,
            )
            self.material_tensors = MaterialUpdateCoefficientTensors(
                ca_er, cb_er, ca_et, cb_et
            )
            self._ca_er, self._cb_er = ca_er, cb_er
            self._ca_et, self._cb_et = ca_et, cb_et
            return
        raise TypeError(
            "material_tensors must be SampledMaterialTensors or "
            "MaterialUpdateCoefficientTensors"
        )

    def _material_tensor_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the accepted staggered shapes for tensor material inputs."""

        if self.config.compress_uniform_material_coefficients:
            return (1, len(self.radii_m)), (1, len(self.radial_midpoints_m))
        return (
            (self.mesh.n_vertices, len(self.radii_m)),
            (self.mesh.n_edges, len(self.radial_midpoints_m)),
        )

    def _validated_material_tensor(
        self,
        values: Any,
        expected_shape: tuple[int, int],
        name: str,
        *,
        minimum: float,
        strict_minimum: bool = False,
        maximum: float | None = None,
    ) -> Any:
        """Normalize one floating Torch tensor without copying it through NumPy."""

        torch = self._runtime.torch
        if not torch.is_tensor(values):
            raise TypeError(f"material_tensors.{name} must be a PyTorch tensor")
        if not torch.is_floating_point(values):
            raise TypeError(
                f"material_tensors.{name} must have a floating-point dtype"
            )
        tensor = self._runtime.as_tensor(values)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"material_tensors.{name} must have shape {expected_shape}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"material_tensors.{name} must be finite")
        below_minimum = tensor <= minimum if strict_minimum else tensor < minimum
        if bool(below_minimum.any()):
            relation = "greater than" if strict_minimum else "at least"
            raise ValueError(
                f"material_tensors.{name} must be {relation} {minimum}"
            )
        if maximum is not None and bool((tensor > maximum).any()):
            raise ValueError(
                f"material_tensors.{name} must be at most {maximum}"
            )
        return tensor

    def _validated_physics_tensor(
        self,
        values: Any,
        expected_shape: tuple[int, ...],
        name: str,
        *,
        minimum: float | None = None,
        strict_minimum: bool = False,
        maximum: float | None = None,
    ) -> Any:
        """Normalize one floating Torch coefficient without severing its graph."""

        torch = self._runtime.torch
        if not torch.is_tensor(values):
            raise TypeError(f"{name} must be a PyTorch tensor")
        if not torch.is_floating_point(values):
            raise TypeError(f"{name} must have a floating-point dtype")
        tensor = self._runtime.as_tensor(values)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
        if minimum is not None:
            below_minimum = (
                tensor <= minimum if strict_minimum else tensor < minimum
            )
            if bool(below_minimum.any()):
                relation = "greater than" if strict_minimum else "at least"
                raise ValueError(f"{name} must be {relation} {minimum}")
        if maximum is not None and bool((tensor > maximum).any()):
            raise ValueError(f"{name} must be at most {maximum}")
        return tensor

    def _torch_exponential_loss_coefficients(
        self, sigma: Any, epsilon: Any
    ) -> tuple[Any, Any]:
        """Differentiably integrate conductive decay, including zero loss."""

        torch = self._runtime.torch
        rate = sigma * self.time_step_s / epsilon
        decay = torch.exp(-rate)
        small = torch.abs(rate) < 1.0e-4
        series = (
            1.0
            - rate / 2.0
            + rate.square() / 6.0
            - rate**3 / 24.0
            + rate**4 / 120.0
        )
        safe_rate = torch.where(small, torch.ones_like(rate), rate)
        regular = -torch.expm1(-safe_rate) / safe_rate
        phi1 = torch.where(small, series, regular)
        drive = self.time_step_s / epsilon * phi1
        return decay, drive

    @staticmethod
    def _uniform_radial_profile(
        values: NDArray[np.float64], label: str
    ) -> NDArray[np.float64]:
        """Return one broadcast row after proving exact horizontal uniformity."""

        profile = values[:1]
        if not np.array_equal(values, np.broadcast_to(profile, values.shape)):
            raise ValueError(
                f"cannot compress horizontally varying {label} coefficients"
            )
        return profile

    def _exponential_loss_coefficients(
        self, sigma: NDArray[np.float64], epsilon: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate conductive decay exactly with midpoint Maxwell forcing."""

        rate = sigma * self.time_step_s / epsilon
        decay = np.exp(-rate)
        phi1 = np.ones_like(rate)
        nonzero = rate != 0.0
        phi1[nonzero] = -np.expm1(-rate[nonzero]) / rate[nonzero]
        drive = self.time_step_s / epsilon * phi1
        return decay, drive

    @staticmethod
    def _validated_material_sample(
        values: tuple[NDArray[np.float64], NDArray[np.float64]],
        expected_shape: tuple[int, int],
        label: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        try:
            sigma_values, epsilon_values = values
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} material sampler must return conductivity and permittivity"
            ) from error
        if any(
            type(value).__module__.startswith("torch")
            for value in (sigma_values, epsilon_values)
        ):
            raise TypeError(
                f"{label} returned PyTorch tensors; pass trainable samples via "
                "material_tensors instead"
            )
        sigma = np.asarray(sigma_values, dtype=np.float64)
        epsilon_r = np.asarray(epsilon_values, dtype=np.float64)
        if sigma.shape != expected_shape or epsilon_r.shape != expected_shape:
            raise ValueError(
                f"{label} material arrays must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(epsilon_r)):
            raise ValueError(f"{label} material arrays must be finite")
        if np.any(sigma < 0.0):
            raise ValueError(f"{label} conductivity cannot be negative")
        if np.any(epsilon_r <= 0.0):
            raise ValueError(f"{label} relative permittivity must be positive")
        return sigma, epsilon_r

    def step(self, count: int = 1, *, currents: Any | None = None) -> None:
        """Advance fields, optionally using backend-native source currents.

        ``currents`` supplies one current value per step. Existing PyTorch
        tensors are moved only when their device or dtype differs, preserving
        their autograd history.
        """

        count = self._validated_count(count, "step count", minimum=0)
        step_currents = self._source_currents(count, currents=currents)
        if self.compiled and count:
            chunk_steps = count - count % self.compile_chunk_size
            for offset in range(0, chunk_steps, self.compile_chunk_size):
                self._field_chunk(
                    step_currents[offset : offset + self.compile_chunk_size]
                )
            for offset in range(chunk_steps, count):
                self._field_step(step_currents[offset])
            self.steps += count
            self.time_s = self.steps * self.time_step_s
            return
        for offset in range(count):
            self._field_step(step_currents[offset])
            self.steps += 1
            self.time_s = self.steps * self.time_step_s

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save a portable, versioned NPZ checkpoint."""

        from .checkpoint import save_checkpoint

        return save_checkpoint(self, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        dtype: str | None = None,
        compile_step: bool = False,
        compile_chunk_size: int = 8,
        torch_threads: int | None = None,
    ) -> GeodesicFDTD:
        """Restore a checkpoint on the selected PyTorch device and dtype."""

        from .checkpoint import load_checkpoint

        return load_checkpoint(
            path,
            device=device,
            dtype=dtype,
            compile_step=compile_step,
            compile_chunk_size=compile_chunk_size,
            torch_threads=torch_threads,
        )

    def _source_currents(
        self, count: int, *, currents: Any | None = None
    ) -> Any:
        if currents is not None:
            if self.source is None:
                raise ValueError("source currents require a configured source")
            values = self._runtime.as_tensor(currents)
            if values.ndim != 1 or values.shape[0] != count:
                raise ValueError(f"currents must have shape ({count},)")
            return values
        if self.source is None:
            return self._runtime.zeros((count,))
        offsets = self._runtime.torch.arange(
            count,
            dtype=self._runtime.dtype,
            device=self._runtime.device,
        )
        times_s = (self.steps + offsets + 0.5) * self.time_step_s
        return self.source.current_tensor_a(times_s, self.time_step_s)

    def _torch_field_state(self) -> Any:
        """Return the current fields as a tensor pytree without copying."""

        return self._field_state_type(self.er, self.et, self.hr, self.ht)

    def _set_torch_field_state(self, state: Any) -> None:
        """Replace wrapper field references after a functional transition."""

        self.er, self.et, self.hr, self.ht = state

    def _torch_optional_physics_state(self) -> Any:
        """Return fields and every ADE state tensor without copying."""

        surface_memory = (
            self._surface_impedance_ade.memory
            if self._surface_impedance_ade is not None
            else None
        )
        plasma_current_density = (
            tuple(self._plasma_coupler.ade.current_density)
            if self._plasma_coupler is not None
            else ()
        )
        return self._optional_physics_state_type(
            self._torch_field_state(),
            surface_memory,
            plasma_current_density,
        )

    def _set_torch_optional_physics_state(self, state: Any) -> None:
        """Replace wrapper field and ADE references after one transition."""

        self._set_torch_field_state(state.fields)
        if self._surface_impedance_ade is not None:
            self._surface_impedance_ade.memory = state.surface_memory
        if self._plasma_coupler is not None:
            self._plasma_coupler.ade.current_density = list(
                state.plasma_current_density
            )

    def _torch_lower_boundary_ht(self, state: Any) -> Any | None:
        if self._surface_impedance_ade is None:
            return None
        lower_surface_gradient = self._lower_surface_gradient_er(
            state, self._field_step_parameters
        )
        boundary_metric = (
            self.radial_midpoints_m[0] / self.radii_m[0]
            if self.config.geometry_mode == "full-spherical"
            else 1.0
        )
        return self._surface_impedance_ade.advance_lower_boundary(
            state.ht[:, 0] + 0.0,
            boundary_metric * state.et[:, 0],
            lower_surface_gradient,
            self._radial_steps[0],
            self.time_step_s,
        )

    def _torch_plasma_currents(
        self, state: Any
    ) -> tuple[Any | None, Any | None]:
        if self._plasma_coupler is None:
            return None, None
        radial, tangential = self._plasma_coupler.advance(state.er, state.et)
        return radial, tangential

    def _advance_torch_fields(self, current_a: Any) -> None:
        if self._has_optional_physics:
            optional_state = self._torch_optional_physics_state()
            stepper = self._tensor_optional_physics_step
            if self._optional_physics_requires_grad(
                optional_state,
                self._optional_physics_step_parameters,
                current_a,
            ):
                stepper = self._tensor_optional_physics_gradient_step
            state = stepper(
                optional_state,
                self._optional_physics_step_parameters,
                current_a,
            )
            self._set_torch_optional_physics_state(state)
            return
        state = self._torch_field_state()
        state = self._tensor_field_step(
            state,
            self._field_step_parameters,
            current_a,
        )
        self._set_torch_field_state(state)

    def _advance_torch_field_chunk(self, currents: Any) -> None:
        """Advance one bare/source or optional-physics functional chunk."""

        if self._has_optional_physics:
            optional_state = self._torch_optional_physics_state()
            stepper = self._tensor_optional_physics_chunk
            if self._optional_physics_requires_grad(
                optional_state,
                self._optional_physics_step_parameters,
                currents,
            ):
                stepper = self._tensor_optional_physics_gradient_chunk
            state = stepper(
                optional_state,
                self._optional_physics_step_parameters,
                currents,
            )
            self._set_torch_optional_physics_state(state)
            return
        self.er, self.et, self.hr, self.ht = self._tensor_field_chunk(
            self._field_state_type(
                self.er, self.et, self.hr, self.ht
            ),
            self._field_step_parameters,
            currents,
        )

    def _advance_fields(self, current_a: Any) -> None:
        self._update_magnetic_fields()
        self._update_electric_fields(current_a)

    def _advance_field_chunk(self, currents: Any) -> None:
        """Advance a fixed-size current chunk inside one compiled graph."""

        for offset in range(self.compile_chunk_size):
            self._advance_fields(currents[offset])

    def _update_magnetic_fields(self) -> None:
        state = self._torch_field_state()
        state = self._advance_magnetic_state(
            state,
            self._field_step_parameters,
            self._torch_lower_boundary_ht(state),
        )
        self._set_torch_field_state(state)

    def _update_electric_fields(self, current_a: Any = 0.0) -> None:
        state = self._torch_field_state()
        radial_plasma, tangential_plasma = self._torch_plasma_currents(state)
        state = self._advance_electric_state(
            state,
            self._field_step_parameters,
            current_a,
            radial_plasma,
            tangential_plasma,
        )
        self._set_torch_field_state(state)

    def diagnostics(self) -> dict[str, float | int | str]:
        """Return inexpensive scalar diagnostics without saving field data."""

        return {
            "step": self.steps,
            "time_s": self.time_s,
            "electric_time_s": self.electric_time_s,
            "magnetic_time_s": self.magnetic_time_s,
            "runtime": "torch",
            "device": str(self._runtime.device),
            "dtype": self._runtime.dtype_name,
            "compiled": self.compiled,
            "compile_chunk_size": self.compile_chunk_size,
            "radial_boundary_condition": self.config.radial_boundary_condition,
            "surface_impedance_terms": (
                self.surface_impedance.terms
                if self.surface_impedance is not None
                else 0
            ),
            "surface_impedance_state_bytes": (
                self._surface_impedance_ade.state_bytes
                if self._surface_impedance_ade is not None
                else 0
            ),
            "plasma_species": len(self.plasma.species) if self.plasma else 0,
            "plasma_state_bytes": (
                self._plasma_coupler.ade.state_bytes
                if self._plasma_coupler is not None
                else 0
            ),
            "loss_integration": self.config.loss_integration,
            "geometry_mode": self.config.geometry_mode,
            "cfl_time_step_limit_s": self.cfl_time_step_limit_s,
            "courant_factor": self.config.courant_factor,
            "field_memory_bytes": self.memory_bytes,
            "persistent_runtime_bytes": self.persistent_runtime_bytes,
            "max_abs_er_v_m": self._runtime.export_max_abs(self.er),
            "max_abs_et_v_m": self._runtime.export_max_abs(self.et),
            "max_abs_hr_a_m": self._runtime.export_max_abs(self.hr),
            "max_abs_ht_a_m": self._runtime.export_max_abs(self.ht),
        }

    @property
    def electric_time_s(self) -> float:
        """Time associated with the integer-step electric fields."""

        return self.steps * self.time_step_s

    @property
    def magnetic_time_s(self) -> float:
        """Time associated with the half-step magnetic fields."""

        return (self.steps - 0.5) * self.time_step_s

    @property
    def runtime(self) -> str:
        """Return constant provenance for the PyTorch-only compute runtime."""

        return "torch"

    @property
    def device(self) -> Any:
        """Return the canonical compute device for this simulation."""

        return self._runtime.device

    @property
    def dtype(self) -> Any:
        """Return the canonical compute dtype for this simulation."""

        return self._runtime.dtype

    @property
    def dtype_name(self) -> str:
        """Return the configured floating-point dtype name."""

        return self._runtime.dtype_name

    @property
    def threads(self) -> int | None:
        """Return the active CPU tensor thread count, when applicable."""

        return self._runtime.threads

    def to_numpy(self, values: Any) -> NDArray[np.generic]:
        """Detach values at a terminal host analysis or plotting boundary."""

        return self._runtime.export_numpy(values)

    def field_value(self, field: str, *indices: int) -> float:
        """Read one field value without exposing backend scalar semantics."""

        try:
            values = getattr(self, field)
        except AttributeError as error:
            raise ValueError("field must be er, et, hr, or ht") from error
        if field not in {"er", "et", "hr", "ht"}:
            raise ValueError("field must be er, et, hr, or ht")
        return self._runtime.export_scalar(values[indices])

    @staticmethod
    def _validated_count(value: int, label: str, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must be an integer")
        result = int(value)
        if result < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{label} must be {qualifier}")
        return result

    @staticmethod
    def _validated_index_array(values: Any, label: str) -> NDArray[np.int64]:
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise ValueError(f"{label} must contain integers")
        return np.asarray(array, dtype=np.int64)

    def record_er_observations(
        self,
        vertex_indices: NDArray[np.int64],
        radial_layers: NDArray[np.int64],
        weights: NDArray[np.float64],
        steps: int,
        *,
        currents: Any | None = None,
        synchronize_every: int | None = None,
    ) -> Any:
        """Advance and return backend-native weighted ``Er`` observations.

        Each row of ``vertex_indices`` and ``weights`` describes one receiver.
        The returned first row is the initial field, followed by one row per
        completed time step.  Keeping the trace buffer on the backend is much
        faster than reading individual MPS or CUDA scalars every step.
        Pass ``synchronize_every`` to request explicit device barriers.
        """

        vertices = self._validated_index_array(vertex_indices, "vertex_indices")
        layers = self._validated_index_array(radial_layers, "radial_layers")
        sample_weights = np.asarray(weights, dtype=np.float64)
        steps = self._validated_count(steps, "step count", minimum=0)
        if synchronize_every is not None:
            synchronize_every = self._validated_count(
                synchronize_every, "synchronize_every", minimum=1
            )
        if vertices.ndim != 2 or sample_weights.shape != vertices.shape:
            raise ValueError("vertex_indices and weights must have matching 2-D shapes")
        if layers.shape != (vertices.shape[0],):
            raise ValueError("radial_layers must contain one layer per observation")
        if np.any(vertices < 0) or np.any(vertices >= self.mesh.n_vertices):
            raise ValueError("observation vertex index is out of range")
        if np.any(layers < 0) or np.any(layers >= len(self.radii_m)):
            raise ValueError("observation radial layer is out of range")
        if not np.all(np.isfinite(sample_weights)):
            raise ValueError("observation weights must be finite")
        if not np.allclose(sample_weights.sum(axis=1), 1.0):
            raise ValueError("observation weights must sum to one")

        backend_vertices = self._runtime.index_tensor(vertices)
        backend_layers = self._runtime.index_tensor(layers)
        backend_weights = self._runtime.as_tensor(sample_weights)
        traces = self._runtime.zeros((steps + 1, vertices.shape[0]))

        def sample(row: int) -> None:
            selected = self.er[backend_vertices, backend_layers[:, None]]
            traces[row] = (selected * backend_weights).sum(axis=1)

        sample(0)
        step_currents = self._source_currents(steps, currents=currents)
        for offset in range(steps):
            self._field_step(step_currents[offset])
            self.steps += 1
            self.time_s = self.steps * self.time_step_s
            sample(offset + 1)
            if synchronize_every and (offset + 1) % synchronize_every == 0:
                self._runtime.synchronize()
        if synchronize_every is not None:
            self._runtime.synchronize()
        return traces

    def record_h_observations(
        self,
        face_indices: NDArray[np.int64],
        face_radial_layers: NDArray[np.int64],
        face_weights: NDArray[np.float64],
        edge_indices: NDArray[np.int64],
        edge_radial_layers: NDArray[np.int64],
        edge_weights: NDArray[np.float64],
        steps: int,
        *,
        currents: Any | None = None,
        synchronize_every: int | None = None,
        sample_every: int = 1,
    ) -> tuple[Any, Any]:
        """Advance while recording weighted radial and tangential H samples.

        The initial row is ``H**(-1/2)`` and row ``k`` is ``H**(k-1/2)``.
        Callers must therefore associate these samples with half-step times,
        independently of the integer-step electric-field clock.
        Pass ``synchronize_every`` to request explicit device barriers.
        """

        faces = self._validated_index_array(face_indices, "face_indices")
        face_layers = self._validated_index_array(
            face_radial_layers, "face_radial_layers"
        )
        radial_weights = np.asarray(face_weights, dtype=np.float64)
        edges = self._validated_index_array(edge_indices, "edge_indices")
        edge_layers = self._validated_index_array(
            edge_radial_layers, "edge_radial_layers"
        )
        tangential_weights = np.asarray(edge_weights, dtype=np.float64)
        steps = self._validated_count(steps, "step count", minimum=0)
        if synchronize_every is not None:
            synchronize_every = self._validated_count(
                synchronize_every, "synchronize_every", minimum=1
            )
        sample_every = self._validated_count(
            sample_every, "sample_every", minimum=1
        )
        if faces.ndim != 2 or radial_weights.shape != faces.shape:
            raise ValueError("face indices and weights must have matching 2-D shapes")
        if edges.ndim != 2 or tangential_weights.shape != edges.shape:
            raise ValueError("edge indices and weights must have matching 2-D shapes")
        if face_layers.shape != faces.shape:
            raise ValueError("face radial layers must match the face indices")
        if edge_layers.shape != edges.shape:
            raise ValueError("edge radial layers must match the edge indices")
        if np.any(faces < 0) or np.any(faces >= self.mesh.n_faces):
            raise ValueError("observation face index is out of range")
        if np.any(edges < 0) or np.any(edges >= self.mesh.n_edges):
            raise ValueError("observation edge index is out of range")
        if np.any(face_layers < 0) or np.any(face_layers >= self.hr.shape[1]):
            raise ValueError("radial H observation layer is out of range")
        if np.any(edge_layers < 0) or np.any(edge_layers >= self.ht.shape[1]):
            raise ValueError("tangential H observation layer is out of range")
        if not np.all(np.isfinite(radial_weights)) or not np.all(
            np.isfinite(tangential_weights)
        ):
            raise ValueError("observation weights must be finite")

        backend_faces = self._runtime.index_tensor(faces)
        backend_face_layers = self._runtime.index_tensor(face_layers)
        backend_radial_weights = self._runtime.as_tensor(radial_weights)
        backend_edges = self._runtime.index_tensor(edges)
        backend_edge_layers = self._runtime.index_tensor(edge_layers)
        backend_tangential_weights = self._runtime.as_tensor(tangential_weights)
        sample_steps = np.concatenate(
            (
                np.arange(0, steps + 1, sample_every, dtype=np.int64),
                np.asarray((steps,), dtype=np.int64),
            )
        )
        sample_steps = np.unique(sample_steps)
        radial_traces = self._runtime.zeros((len(sample_steps), faces.shape[0]))
        tangential_traces = self._runtime.zeros((len(sample_steps), edges.shape[0]))
        step_currents = self._source_currents(steps, currents=currents)

        def sample(row: int) -> None:
            selected_hr = self.hr[backend_faces, backend_face_layers]
            radial_traces[row] = (
                selected_hr * backend_radial_weights
            ).sum(axis=1)
            selected_ht = self.ht[backend_edges, backend_edge_layers]
            tangential_traces[row] = (
                selected_ht * backend_tangential_weights
            ).sum(axis=1)

        sample(0)
        previous_step = 0
        for row, target_step in enumerate(sample_steps[1:], start=1):
            target = int(target_step)
            self.step(
                target - previous_step,
                currents=(
                    step_currents[previous_step:target]
                    if self.source is not None
                    else None
                ),
            )
            sample(row)
            previous_step = target
            if synchronize_every and target % synchronize_every == 0:
                self._runtime.synchronize()
        if synchronize_every is not None:
            self._runtime.synchronize()
        return radial_traces, tangential_traces

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the four evolving field arrays."""

        return sum(
            self._runtime.nbytes(field)
            for field in (self.er, self.et, self.hr, self.ht)
        )

    @property
    def persistent_runtime_bytes(self) -> int:
        """Bytes in persistent field, coefficient, metric, and topology tensors."""

        solver_names = (
            "er",
            "et",
            "hr",
            "ht",
            "_ca_er",
            "_cb_er",
            "_ca_et",
            "_cb_et",
            "_primal_edge_angles",
            "_inverse_primal_edge_angles",
            "_dual_edge_angles",
            "_inverse_dual_edge_angles",
            "_inverse_dual_cell_solid_angles",
            "_inverse_face_solid_angles",
            "_radii",
            "_inverse_radii",
            "_radial_midpoints",
            "_inverse_radial_midpoints",
            "_radial_steps",
            "_radial_node_control_lengths",
            "_radial_center_distances",
        )
        backend_names = (
            "edges",
            "face_edges",
            "face_edge_signs",
            "edge_left_faces",
            "edge_right_faces",
            "vertex_edges",
            "vertex_edge_signs",
        )
        arrays = [getattr(self, name) for name in solver_names]
        arrays.extend(
            getattr(self._runtime, name)
            for name in backend_names
            if hasattr(self._runtime, name)
        )
        for distribution in (
            self._source_distribution,
            self._tangential_source_distribution,
        ):
            if distribution is not None:
                arrays.extend(distribution)
        unique = {id(array): array for array in arrays}
        total = sum(self._runtime.nbytes(array) for array in unique.values())
        if self._surface_impedance_ade is not None:
            total += self._surface_impedance_ade.persistent_bytes
        if self._plasma_coupler is not None:
            total += self._plasma_coupler.persistent_bytes
        return total


class _StaticPreparationContext:
    """Lightweight host-only context for shared static preparation."""

    _sample_material_properties = GeodesicFDTD._sample_material_properties
    _dual_cell_material_average = GeodesicFDTD._dual_cell_material_average
    _validated_material_sample = staticmethod(
        GeodesicFDTD._validated_material_sample
    )
    _estimate_cfl_time_step_limit = GeodesicFDTD._estimate_cfl_time_step_limit


def _prepare_static_simulation(
    config: SimulationConfig,
    mesh: GeodesicMesh,
    material: EarthIonosphereMaterial,
    source: GaussianCurrent | TangentialGaussianCurrent | None,
    plasma: MeshPlasmaModel | None,
) -> _StaticSimulationData:
    """Prepare host metadata without allocating evolving field arrays."""

    context = _StaticPreparationContext()
    context.config = config
    context.mesh = mesh
    context.material = material
    context.source = source
    context.plasma = plasma
    if config.radial_altitudes_m is None:
        context.altitudes_m = np.linspace(
            config.minimum_altitude_m,
            config.maximum_altitude_m,
            config.radial_cells + 1,
        )
    else:
        context.altitudes_m = np.asarray(
            config.radial_altitudes_m, dtype=np.float64
        )
    context.radii_m = config.earth_radius_m + context.altitudes_m
    context.radial_midpoints_m = 0.5 * (
        context.radii_m[:-1] + context.radii_m[1:]
    )
    context.radial_midpoint_altitudes_m = (
        context.radial_midpoints_m - config.earth_radius_m
    )
    context.radial_steps_m = np.diff(context.radii_m)
    context.radial_node_control_lengths_m = np.empty(
        len(context.radii_m), dtype=np.float64
    )
    context.radial_node_control_lengths_m[0] = (
        0.5 * context.radial_steps_m[0]
    )
    context.radial_node_control_lengths_m[-1] = (
        0.5 * context.radial_steps_m[-1]
    )
    context.radial_node_control_lengths_m[1:-1] = np.diff(
        context.radial_midpoints_m
    )
    if plasma is not None:
        plasma.validate_grid(mesh, context.radial_midpoint_altitudes_m)

    context._sample_material_properties()
    cfl_time_step_limit_s = context._estimate_cfl_time_step_limit()
    maximum_stable_time_step_s = (
        config.courant_factor * cfl_time_step_limit_s
    )
    time_step_s = (
        config.time_step_s
        if config.time_step_s is not None
        else maximum_stable_time_step_s
    )
    if time_step_s > maximum_stable_time_step_s * (1.0 + 1.0e-12):
        raise ValueError(
            f"time step {time_step_s:.6e} s exceeds conservative limit "
            f"{maximum_stable_time_step_s:.6e} s"
        )
    if (
        source is not None
        and source.carrier_frequency_hz >= 0.5 / time_step_s
    ):
        raise ValueError(
            "source carrier frequency must be below the time-step Nyquist limit"
        )

    source_distribution = None
    tangential_source_distribution = None
    if isinstance(source, TangentialGaussianCurrent):
        tangential_source_distribution = source.edge_distribution(context)
    elif source is not None:
        source_distribution = source.staggered_distribution(context)

    return _StaticSimulationData(
        altitudes_m=context.altitudes_m,
        radii_m=context.radii_m,
        radial_midpoints_m=context.radial_midpoints_m,
        radial_midpoint_altitudes_m=context.radial_midpoint_altitudes_m,
        radial_steps_m=context.radial_steps_m,
        radial_node_control_lengths_m=context.radial_node_control_lengths_m,
        sigma_er=context.sigma_er,
        epsilon_r_er=context.epsilon_r_er,
        sigma_et=context.sigma_et,
        epsilon_r_et=context.epsilon_r_et,
        cfl_time_step_limit_s=cfl_time_step_limit_s,
        maximum_stable_time_step_s=maximum_stable_time_step_s,
        time_step_s=time_step_s,
        source_distribution=source_distribution,
        tangential_source_distribution=tangential_source_distribution,
        anomaly_horizontal_fractions_er=getattr(
            context, "anomaly_horizontal_fractions_er", None
        ),
        anomaly_horizontal_fractions_et=getattr(
            context, "anomaly_horizontal_fractions_et", None
        ),
    )


def _host_material_update_coefficients(
    data: _StaticSimulationData,
    config: SimulationConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return static lossy update coefficients without a compute backend."""

    epsilon_er = EPSILON_0 * data.epsilon_r_er
    epsilon_et = EPSILON_0 * data.epsilon_r_et
    if config.loss_integration == "trapezoidal":
        loss_er = data.sigma_er * data.time_step_s / (2.0 * epsilon_er)
        loss_et = data.sigma_et * data.time_step_s / (2.0 * epsilon_et)
        ca_er = (1.0 - loss_er) / (1.0 + loss_er)
        cb_er = data.time_step_s / (epsilon_er * (1.0 + loss_er))
        ca_et = (1.0 - loss_et) / (1.0 + loss_et)
        cb_et = data.time_step_s / (epsilon_et * (1.0 + loss_et))
    else:
        ca_er, cb_er = _host_exponential_loss_coefficients(
            data.sigma_er, epsilon_er, data.time_step_s
        )
        ca_et, cb_et = _host_exponential_loss_coefficients(
            data.sigma_et, epsilon_et, data.time_step_s
        )
    if config.compress_uniform_material_coefficients:
        ca_er = _uniform_radial_profile(ca_er, "radial electric")
        cb_er = _uniform_radial_profile(cb_er, "radial electric")
        ca_et = _uniform_radial_profile(ca_et, "tangential electric")
        cb_et = _uniform_radial_profile(cb_et, "tangential electric")
    return ca_er, cb_er, ca_et, cb_et


def _host_exponential_loss_coefficients(
    sigma: NDArray[np.float64],
    epsilon: NDArray[np.float64],
    time_step_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate static conductive decay exactly on the host."""

    rate = sigma * time_step_s / epsilon
    decay = np.exp(-rate)
    phi1 = np.ones_like(rate)
    nonzero = rate != 0.0
    phi1[nonzero] = -np.expm1(-rate[nonzero]) / rate[nonzero]
    drive = time_step_s / epsilon * phi1
    return decay, drive


def _uniform_radial_profile(
    values: NDArray[np.float64], label: str
) -> NDArray[np.float64]:
    """Return one row after proving exact horizontal uniformity."""

    profile = values[:1]
    if not np.array_equal(values, np.broadcast_to(profile, values.shape)):
        raise ValueError(
            f"cannot compress horizontally varying {label} coefficients"
        )
    return profile

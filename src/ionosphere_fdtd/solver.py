"""Vectorized 3-D geodesic FDTD time stepping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backends import ArrayBackend, create_backend
from .constants import C_0, EARTH_RADIUS_M, EPSILON_0, MU_0
from .materials import (
    EarthIonosphereMaterial,
    apply_fractional_cell_anomalies,
    apply_fractional_point_anomalies,
    conservative_anomaly_fractions,
)
from .mesh import GeodesicMesh, build_geodesic_mesh
from .plasma import GeodesicPlasmaCoupler, MeshPlasmaModel
from .radial_grid import validate_radial_grid
from .sources import GaussianCurrent, TangentialGaussianCurrent
from .surface_impedance import (
    ConductiveHalfSpaceSurface,
    SurfaceImpedanceADE,
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
        backend: str = "numpy",
        device: str = "auto",
        dtype: str = "auto",
        compile_step: bool = False,
        compile_chunk_size: int = 8,
        torch_threads: int | None = None,
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
        self.backend: ArrayBackend = create_backend(
            backend,
            self.mesh,
            device=device,
            dtype=dtype,
            torch_threads=torch_threads,
        )
        self.material = material or EarthIonosphereMaterial()
        self.source = source
        self.surface_impedance = surface_impedance
        self.plasma = plasma
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

        if self.config.radial_altitudes_m is None:
            self.altitudes_m = np.linspace(
                self.config.minimum_altitude_m,
                self.config.maximum_altitude_m,
                self.config.radial_cells + 1,
            )
        else:
            self.altitudes_m = np.asarray(
                self.config.radial_altitudes_m, dtype=np.float64
            )
        self.radii_m = self.config.earth_radius_m + self.altitudes_m
        self.radial_midpoints_m = 0.5 * (self.radii_m[:-1] + self.radii_m[1:])
        self.radial_midpoint_altitudes_m = (
            self.radial_midpoints_m - self.config.earth_radius_m
        )
        self.radial_steps_m = np.diff(self.radii_m)
        self.radial_node_control_lengths_m = np.empty(
            len(self.radii_m), dtype=np.float64
        )
        self.radial_node_control_lengths_m[0] = 0.5 * self.radial_steps_m[0]
        self.radial_node_control_lengths_m[-1] = 0.5 * self.radial_steps_m[-1]
        self.radial_node_control_lengths_m[1:-1] = np.diff(
            self.radial_midpoints_m
        )
        if plasma is not None:
            plasma.validate_grid(self.mesh, self.radial_midpoint_altitudes_m)

        # Material permittivity controls the fastest supported wave speed, so it
        # must be known before an automatic or user-supplied time step is
        # validated.  Keep sampling separate from coefficient construction:
        # the latter depends on dt, while the former does not.
        self._sample_material_properties()
        self.cfl_time_step_limit_s = self._estimate_cfl_time_step_limit()
        self.maximum_stable_time_step_s = (
            self.config.courant_factor * self.cfl_time_step_limit_s
        )
        self.time_step_s = (
            self.config.time_step_s
            if self.config.time_step_s is not None
            else self.maximum_stable_time_step_s
        )
        if self.time_step_s > self.maximum_stable_time_step_s * (1.0 + 1.0e-12):
            raise ValueError(
                f"time step {self.time_step_s:.6e} s exceeds conservative limit "
                f"{self.maximum_stable_time_step_s:.6e} s"
            )
        if (
            self.source is not None
            and self.source.carrier_frequency_hz
            >= 0.5 / self.time_step_s
        ):
            raise ValueError(
                "source carrier frequency must be below the time-step Nyquist limit"
            )

        self.er = self.backend.zeros((self.mesh.n_vertices, len(self.radii_m)))
        self.ht = self.backend.zeros((self.mesh.n_edges, len(self.radii_m)))
        self.et = self.backend.zeros(
            (self.mesh.n_edges, len(self.radial_midpoints_m))
        )
        self.hr = self.backend.zeros(
            (self.mesh.n_faces, len(self.radial_midpoints_m))
        )
        self._surface_impedance_ade = (
            SurfaceImpedanceADE(
                surface_impedance,
                edge_count=self.mesh.n_edges,
                time_step_s=self.time_step_s,
                backend=self.backend,
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
                self.backend,
            )
            if plasma is not None
            else None
        )

        self._prepare_geometry()
        self._prepare_material_coefficients()
        self._source_distribution = None
        self._tangential_source_distribution = None
        if isinstance(self.source, TangentialGaussianCurrent):
            edges, layers, weights = self.source.edge_distribution(self)
            self._tangential_source_distribution = (
                self.backend.index_array(edges),
                self.backend.index_array(layers),
                self.backend.asarray(weights),
            )
        elif self.source is not None:
            vertices, layers, weights = self.source.staggered_distribution(self)
            self._source_distribution = (
                self.backend.index_array(vertices),
                self.backend.index_array(layers),
                self.backend.asarray(weights),
            )
        self.time_s = 0.0
        self.steps = 0
        self.compiled = compile_step
        self._field_step = (
            self.backend.compile_step(self._advance_fields)
            if compile_step
            else self._advance_fields
        )
        self._field_chunk = (
            self.backend.compile_step(self._advance_field_chunk)
            if compile_step
            else self._advance_field_chunk
        )

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
        self._primal_edge_angles = self.backend.asarray(
            self.mesh.primal_edge_angles[:, None]
        )
        self._inverse_primal_edge_angles = self.backend.asarray(
            1.0 / self.mesh.primal_edge_angles[:, None]
        )
        self._dual_edge_angles = self.backend.asarray(
            self.mesh.dual_edge_angles[:, None]
        )
        self._inverse_dual_edge_angles = self.backend.asarray(
            1.0 / self.mesh.dual_edge_angles[:, None]
        )
        self._inverse_dual_cell_solid_angles = self.backend.asarray(
            1.0 / self.mesh.dual_cell_solid_angles[:, None]
        )
        self._inverse_face_solid_angles = self.backend.asarray(
            1.0 / self.mesh.face_solid_angles[:, None]
        )
        self._radii = self.backend.asarray(self.radii_m)
        self._inverse_radii = self.backend.asarray(1.0 / self.radii_m[None, :])
        self._radial_midpoints = self.backend.asarray(self.radial_midpoints_m)
        self._inverse_radial_midpoints = self.backend.asarray(
            1.0 / self.radial_midpoints_m[None, :]
        )
        self._radial_steps = self.backend.asarray(self.radial_steps_m)
        self._radial_node_control_lengths = self.backend.asarray(
            self.radial_node_control_lengths_m
        )
        self._radial_center_distances = self.backend.asarray(
            self.radial_midpoints_m[1:] - self.radial_midpoints_m[:-1]
        )
    def _radial_derivative_et(self) -> Any:
        values = self.et
        if self.config.geometry_mode == "full-spherical":
            values = values * self._radial_midpoints[None, :]
        result = self.backend.empty_like(self.ht)
        result[:, 0] = 2.0 * values[:, 0] / self._radial_steps[0]
        result[:, -1] = -2.0 * values[:, -1] / self._radial_steps[-1]
        if self.ht.shape[1] > 2:
            result[:, 1:-1] = self.backend.diff(
                values, axis=1
            ) / self._radial_center_distances
        if self.config.geometry_mode == "full-spherical":
            result *= self._inverse_radii
        return result

    def _radial_derivative_ht(self) -> Any:
        values = self.ht
        if self.config.geometry_mode == "full-spherical":
            values = values * self._radii[None, :]
        result = self.backend.diff(values, axis=1) / self._radial_steps[None, :]
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

    def _prepare_material_coefficients(self) -> None:
        """Build lossy electric-field update coefficients at the selected dt."""

        epsilon_er = EPSILON_0 * self.epsilon_r_er
        epsilon_et = EPSILON_0 * self.epsilon_r_et
        sigma_er = self.sigma_er
        sigma_et = self.sigma_et
        if self.config.loss_integration == "trapezoidal":
            loss_er = sigma_er * self.time_step_s / (2.0 * epsilon_er)
            loss_et = sigma_et * self.time_step_s / (2.0 * epsilon_et)
            ca_er = (1.0 - loss_er) / (1.0 + loss_er)
            cb_er = self.time_step_s / (epsilon_er * (1.0 + loss_er))
            ca_et = (1.0 - loss_et) / (1.0 + loss_et)
            cb_et = self.time_step_s / (epsilon_et * (1.0 + loss_et))
        else:
            ca_er, cb_er = self._exponential_loss_coefficients(
                sigma_er, epsilon_er
            )
            ca_et, cb_et = self._exponential_loss_coefficients(
                sigma_et, epsilon_et
            )
        if self.config.compress_uniform_material_coefficients:
            ca_er = self._uniform_radial_profile(ca_er, "radial electric")
            cb_er = self._uniform_radial_profile(cb_er, "radial electric")
            ca_et = self._uniform_radial_profile(ca_et, "tangential electric")
            cb_et = self._uniform_radial_profile(cb_et, "tangential electric")
        self._ca_er = self.backend.asarray(ca_er)
        self._cb_er = self.backend.asarray(cb_er)
        self._ca_et = self.backend.asarray(ca_et)
        self._cb_et = self.backend.asarray(cb_et)

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

    def step(self, count: int = 1) -> None:
        """Advance the fields by ``count`` complete leapfrog time steps."""

        count = self._validated_count(count, "step count", minimum=0)
        if self.compiled and count:
            currents = self._source_currents(count)
            chunk_steps = count - count % self.compile_chunk_size
            for offset in range(0, chunk_steps, self.compile_chunk_size):
                self._field_chunk(
                    currents[offset : offset + self.compile_chunk_size]
                )
            for offset in range(chunk_steps, count):
                self._field_step(currents[offset])
            self.steps += count
            self.time_s = self.steps * self.time_step_s
            return
        for _ in range(count):
            current_a = (
                self.source.current_a(
                    self.time_s + 0.5 * self.time_step_s,
                    self.time_step_s,
                )
                if self.source is not None
                else 0.0
            )
            self._field_step(current_a)
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
        backend: str = "numpy",
        device: str = "auto",
        dtype: str | None = None,
        compile_step: bool = False,
        compile_chunk_size: int = 8,
        torch_threads: int | None = None,
    ) -> GeodesicFDTD:
        """Restore a checkpoint, optionally on a different backend or device."""

        from .checkpoint import load_checkpoint

        return load_checkpoint(
            path,
            backend=backend,
            device=device,
            dtype=dtype,
            compile_step=compile_step,
            compile_chunk_size=compile_chunk_size,
            torch_threads=torch_threads,
        )

    def _source_currents(self, count: int) -> Any:
        if self.source is None:
            return self.backend.zeros((count,))
        values = np.fromiter(
            (
                self.source.current_a(
                    (self.steps + offset + 0.5) * self.time_step_s,
                    self.time_step_s,
                )
                for offset in range(count)
            ),
            dtype=np.float64,
            count=count,
        )
        return self.backend.asarray(values)

    def _advance_fields(self, current_a: Any) -> None:
        self._update_magnetic_fields()
        self._update_electric_fields(current_a)

    def _advance_field_chunk(self, currents: Any) -> None:
        """Advance a fixed-size current chunk inside one compiled graph."""

        for offset in range(self.compile_chunk_size):
            self._advance_fields(currents[offset])

    def _update_magnetic_fields(self) -> None:
        surface_gradient_er = self.backend.edge_difference(self.er)
        surface_gradient_er *= self._inverse_primal_edge_angles
        surface_gradient_er *= self._inverse_radii
        lower_surface_gradient = (
            surface_gradient_er[:, 0] + 0.0
            if self._surface_impedance_ade is not None
            else None
        )
        lower_h_old = (
            self.ht[:, 0] + 0.0
            if self._surface_impedance_ade is not None
            else None
        )

        # The default odd Et ghosts place zero tangential electric field at
        # both radial boundaries. A lower surface ADE overwrites its Ht update
        # below; the upper boundary remains PEC.
        radial_derivative_et = self._radial_derivative_et()

        surface_gradient_er -= radial_derivative_et
        surface_gradient_er *= self.time_step_s / MU_0
        self.ht += surface_gradient_er
        if self._surface_impedance_ade is not None:
            boundary_metric = (
                self.radial_midpoints_m[0] / self.radii_m[0]
                if self.config.geometry_mode == "full-spherical"
                else 1.0
            )
            self.ht[:, 0] = self._surface_impedance_ade.advance_lower_boundary(
                lower_h_old,
                boundary_metric * self.et[:, 0],
                lower_surface_gradient,
                self._radial_steps[0],
                self.time_step_s,
            )
        del surface_gradient_er, radial_derivative_et

        electric_circulation = self.backend.face_circulation(
            self.et * self._primal_edge_angles
        )
        electric_circulation *= self._inverse_face_solid_angles
        electric_circulation *= self._inverse_radial_midpoints
        electric_circulation *= self.time_step_s / MU_0
        self.hr -= electric_circulation

    def _update_electric_fields(self, current_a: Any = 0.0) -> None:
        plasma_current = (
            self._plasma_coupler.advance(self.er, self.et)
            if self._plasma_coupler is not None
            else None
        )
        magnetic_circulation = self.backend.dual_cell_circulation(
            self.ht * self._dual_edge_angles
        )
        magnetic_circulation *= self._inverse_dual_cell_solid_angles
        magnetic_circulation *= self._inverse_radii

        current_density = None
        if self.source is not None and self._source_distribution is not None:
            vertices, layers, weights = self._source_distribution
            current_density = (
                weights
                * current_a
                * self.source.vertical_element_length_m
                * self._inverse_dual_cell_solid_angles[vertices, 0]
                / self._radii[layers] ** 2
                / self._radial_node_control_lengths[layers]
            )

        self.er *= self._ca_er
        magnetic_circulation *= self._cb_er
        self.er += magnetic_circulation
        if current_density is not None:
            vertices, layers, _ = self._source_distribution
            coefficient_vertices = (
                0
                if self.config.compress_uniform_material_coefficients
                else vertices
            )
            self.er[vertices, layers] -= (
                self._cb_er[coefficient_vertices, layers] * current_density
            )
        if plasma_current is not None:
            radial_plasma_current, tangential_plasma_current = plasma_current
            self.er -= self._cb_er * radial_plasma_current

        surface_gradient_hr = self.backend.dual_edge_difference(self.hr)
        surface_gradient_hr *= self._inverse_dual_edge_angles
        surface_gradient_hr *= self._inverse_radial_midpoints
        radial_derivative_ht = self._radial_derivative_ht()
        surface_gradient_hr -= radial_derivative_ht
        del radial_derivative_ht
        self.et *= self._ca_et
        surface_gradient_hr *= self._cb_et
        self.et += surface_gradient_hr
        if self._tangential_source_distribution is not None:
            edges, layers, weights = self._tangential_source_distribution
            current_density = (
                weights
                * current_a
                * self._inverse_dual_edge_angles[edges, 0]
                / self._radial_midpoints[layers]
                / self._radial_steps[layers]
            )
            coefficient_edges = (
                0 if self.config.compress_uniform_material_coefficients else edges
            )
            self.et[edges, layers] -= (
                self._cb_et[coefficient_edges, layers] * current_density
            )
        if plasma_current is not None:
            self.et -= self._cb_et * tangential_plasma_current

    def diagnostics(self) -> dict[str, float | int | str]:
        """Return inexpensive scalar diagnostics without saving field data."""

        return {
            "step": self.steps,
            "time_s": self.time_s,
            "electric_time_s": self.electric_time_s,
            "magnetic_time_s": self.magnetic_time_s,
            "backend": self.backend.name,
            "device": self.backend.device,
            "dtype": self.backend.dtype_name,
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
            "persistent_backend_bytes": self.persistent_backend_bytes,
            "max_abs_er_v_m": self.backend.max_abs(self.er),
            "max_abs_et_v_m": self.backend.max_abs(self.et),
            "max_abs_hr_a_m": self.backend.max_abs(self.hr),
            "max_abs_ht_a_m": self.backend.max_abs(self.ht),
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
    def device(self) -> Any:
        """Return the canonical compute device for this simulation."""

        runtime = getattr(self.backend, "_runtime", None)
        return runtime.device if runtime is not None else self.backend.device

    @property
    def dtype(self) -> Any:
        """Return the canonical compute dtype for this simulation."""

        return self.backend.dtype

    @property
    def threads(self) -> int | None:
        """Return the active CPU tensor thread count, when applicable."""

        return self.backend.threads

    def to_numpy(self, values: Any) -> NDArray[np.generic]:
        """Detach values at a terminal host analysis or plotting boundary."""

        return self.backend.to_numpy(values)

    def field_value(self, field: str, *indices: int) -> float:
        """Read one field value without exposing backend scalar semantics."""

        try:
            values = getattr(self, field)
        except AttributeError as error:
            raise ValueError("field must be er, et, hr, or ht") from error
        if field not in {"er", "et", "hr", "ht"}:
            raise ValueError("field must be er, et, hr, or ht")
        return self.backend.scalar(values[indices])

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
        synchronize_every: int = 128,
    ) -> NDArray[np.generic]:
        """Advance and record weighted ``Er`` observations without host syncs.

        Each row of ``vertex_indices`` and ``weights`` describes one receiver.
        The returned first row is the initial field, followed by one row per
        completed time step.  Keeping the trace buffer on the backend is much
        faster than reading individual MPS or CUDA scalars every step.
        """

        vertices = self._validated_index_array(vertex_indices, "vertex_indices")
        layers = self._validated_index_array(radial_layers, "radial_layers")
        sample_weights = np.asarray(weights, dtype=np.float64)
        steps = self._validated_count(steps, "step count", minimum=0)
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

        backend_vertices = self.backend.index_array(vertices)
        backend_layers = self.backend.index_array(layers)
        backend_weights = self.backend.asarray(sample_weights)
        traces = self.backend.zeros((steps + 1, vertices.shape[0]))

        def sample(row: int) -> None:
            selected = self.er[backend_vertices, backend_layers[:, None]]
            traces[row] = (selected * backend_weights).sum(axis=1)

        sample(0)
        currents = self._source_currents(steps)
        for offset in range(steps):
            self._field_step(currents[offset])
            self.steps += 1
            self.time_s = self.steps * self.time_step_s
            sample(offset + 1)
            if (offset + 1) % synchronize_every == 0:
                self.backend.synchronize()
        self.backend.synchronize()
        return self.to_numpy(traces)

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
        synchronize_every: int = 128,
        sample_every: int = 1,
    ) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
        """Advance while recording weighted radial and tangential H samples.

        The initial row is ``H**(-1/2)`` and row ``k`` is ``H**(k-1/2)``.
        Callers must therefore associate these samples with half-step times,
        independently of the integer-step electric-field clock.
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

        backend_faces = self.backend.index_array(faces)
        backend_face_layers = self.backend.index_array(face_layers)
        backend_radial_weights = self.backend.asarray(radial_weights)
        backend_edges = self.backend.index_array(edges)
        backend_edge_layers = self.backend.index_array(edge_layers)
        backend_tangential_weights = self.backend.asarray(tangential_weights)
        sample_steps = np.concatenate(
            (
                np.arange(0, steps + 1, sample_every, dtype=np.int64),
                np.asarray((steps,), dtype=np.int64),
            )
        )
        sample_steps = np.unique(sample_steps)
        radial_traces = self.backend.zeros((len(sample_steps), faces.shape[0]))
        tangential_traces = self.backend.zeros((len(sample_steps), edges.shape[0]))

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
        for row, target_step in enumerate(sample_steps[1:], start=1):
            self.step(int(target_step) - self.steps)
            sample(row)
            if int(target_step) % synchronize_every == 0:
                self.backend.synchronize()
        self.backend.synchronize()
        return self.to_numpy(radial_traces), self.to_numpy(tangential_traces)

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the four evolving field arrays."""

        return sum(
            self.backend.nbytes(field)
            for field in (self.er, self.et, self.hr, self.ht)
        )

    @property
    def persistent_backend_bytes(self) -> int:
        """Bytes in persistent field, coefficient, metric, and topology arrays."""

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
            getattr(self.backend, name)
            for name in backend_names
            if hasattr(self.backend, name)
        )
        for distribution in (
            self._source_distribution,
            self._tangential_source_distribution,
        ):
            if distribution is not None:
                arrays.extend(distribution)
        unique = {id(array): array for array in arrays}
        total = sum(self.backend.nbytes(array) for array in unique.values())
        if self._surface_impedance_ade is not None:
            total += self._surface_impedance_ade.persistent_bytes
        if self._plasma_coupler is not None:
            total += self._plasma_coupler.persistent_bytes
        return total

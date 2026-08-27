"""Reproduce Simpson, Heikes, and Taflove (2006), Figures 5--7."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ionosphere_fdtd.adaptive_mesh import (
    SphericalRefinementRegion,
    build_adaptive_geodesic_mesh,
)
from ionosphere_fdtd.materials import SphericalAnomaly
from ionosphere_fdtd.mesh import GeodesicMesh
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import (
    TangentialGaussianCurrent,
    geographic_direction,
    geographic_face_index,
    geographic_tangent_basis,
    radial_linear_distribution,
)

from ..common.archive import save_npz_atomic
from ..simpson_taflove_2004.materials import (
    ETOPO5_SHA256,
    ETOPO5Relief,
    SimpsonTaflove2004Material,
)
from ..simpson_taflove_2004.model import (
    REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M,
    AttenuationCurves,
    ValidationTraces,
    bannister_figure_8_guide,
    natural_earth_land_classifier,
)
from ..simpson_taflove_2004.model import (
    render_receiver_grid as _render_receiver_grid,
)
from .materials import HermanceFigure15Material

FloatArray = NDArray[np.float64]

PAPER_SUBDIVISION = 7
PAPER_SURFACE_CELLS = 163_842
PAPER_NOMINAL_RADIAL_CELLS = 40
PAPER_RADIAL_SPACING_M = 5_000.0
PAPER_SUBGRID_SPACING_M = 1_250.0
PAPER_TRANSMITTER_LATITUDE_DEG = 46.5
PAPER_TRANSMITTER_LONGITUDE_DEG = -90.9
PAPER_TRANSMITTER_LINE_LENGTH_M = 22_500.0
PAPER_TRANSMITTER_CURRENT_A = 300.0
PAPER_CARRIER_FREQUENCY_HZ = 20.0
PAPER_ENVELOPE_FWHM_S = 42.5e-3
PAPER_ENVELOPE_ONE_OVER_E_HALF_WIDTH_S = PAPER_ENVELOPE_FWHM_S / (
    2.0 * np.sqrt(np.log(2.0))
)
PAPER_SOURCE_CENTER_S = 3.0 * PAPER_ENVELOPE_ONE_OVER_E_HALF_WIDTH_S
PAPER_OIL_LATITUDE_DEG = 69.0
PAPER_OIL_LONGITUDE_DEG = -156.0
PAPER_OIL_AREA_KM2 = 4_800.0
PAPER_OIL_RADIUS_M = 1_000.0 * np.sqrt(PAPER_OIL_AREA_KM2 / np.pi)
PAPER_OIL_THICKNESS_M = 1_250.0
PAPER_OIL_MEDIAN_DEPTH_M = 1_200.0
PAPER_OIL_CONDUCTIVITY_FACTOR = 0.1
PAPER_FIGURE_7_DURATION_S = 0.085
PAPER_ADAPTIVE_BASE_SUBDIVISION = 7
PAPER_ADAPTIVE_TARGET_SUBDIVISIONS = (9, 10)
PAPER_ADAPTIVE_CORE_RADIUS_DEG = 1.0
PAPER_ADAPTIVE_TRANSITION_WIDTH_DEG = 1.0
PAPER_DAYTIME_IONOSPHERE_REFERENCE_HEIGHT_M = 70_000.0
PAPER_DAYTIME_IONOSPHERE_SCALE_HEIGHT_M = 1_000.0 / 0.3
THESIS_NIGHTTIME_IONOSPHERE_REFERENCE_HEIGHT_M = 92_800.0
THESIS_NIGHTTIME_IONOSPHERE_SCALE_HEIGHT_M = 2_470.0
THESIS_DAYTIME_EFFECTIVE_REFLECTION_HEIGHT_M = 48_000.0
THESIS_NIGHTTIME_EFFECTIVE_REFLECTION_HEIGHT_M = 76_000.0
THESIS_DAWN_LONGITUDE_DEG = 0.0
THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG = THESIS_DAWN_LONGITUDE_DEG + 90.0
THESIS_FIGURE_15_SHALLOW_RESISTIVITY_LIMIT_OHM_M = 10.0
THESIS_FIGURE_15_CONTINENTAL_RESISTIVITY_LIMIT_OHM_M = 5_000.0
THESIS_FIGURE_15_DEEP_RESISTIVITY_LIMIT_OHM_M = 50.0
THESIS_OIL_MAXIMUM_BACKGROUND_CONDUCTIVITY_S_M = 0.2


@dataclass(frozen=True, slots=True)
class DayNightHemisphereProfile:
    """Select day/night profile values across a declared solar terminator."""

    daytime_value: float
    nighttime_value: float
    subsolar_latitude_deg: float = 0.0
    subsolar_longitude_deg: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.daytime_value,
            self.nighttime_value,
            self.subsolar_latitude_deg,
            self.subsolar_longitude_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("day/night profile values must be finite")
        if not -90.0 <= self.subsolar_latitude_deg <= 90.0:
            raise ValueError("subsolar latitude must be in [-90, 90]")

    def __call__(self, directions: FloatArray) -> FloatArray:
        points = np.asarray(directions, dtype=np.float64)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        sunward = geographic_direction(
            self.subsolar_latitude_deg, self.subsolar_longitude_deg
        )
        return np.where(
            points @ sunward >= 0.0,
            self.daytime_value,
            self.nighttime_value,
        )


def render_receiver_grid(
    output: str | Path, *, display_subdivision: int = 4
) -> Path:
    """Render the Figure 5--6 source and receivers on the production topology."""

    return _render_receiver_grid(
        output,
        display_subdivision=display_subdivision,
        production_subdivision=PAPER_SUBDIVISION,
        study_label="Simpson, Heikes, and Taflove (2006)",
    )


@dataclass(frozen=True, slots=True)
class RadarTraces:
    """Surface magnetic-field observations for one Figure 7 model."""

    time_s: FloatArray
    hr_a_m: FloatArray
    ht_east_a_m: FloatArray
    ht_north_a_m: FloatArray
    source_center_s: float
    case: str
    run_signature: str


@dataclass(frozen=True, slots=True)
class RadarPerturbation:
    """Pointwise normalized magnetic perturbations used in Figure 7."""

    time_s: FloatArray
    delta_ht_db: FloatArray
    delta_hr_db: FloatArray
    valid_ht: NDArray[np.bool_]
    valid_hr: NDArray[np.bool_]
    ht_projection_east_north: FloatArray
    normalization: str
    ht_definition: str


@dataclass(frozen=True, slots=True)
class RadarResolutionConvergence:
    """Relative L2 changes between two adaptive radar resolutions."""

    coarse_target_subdivision: int
    fine_target_subdivision: int
    comparison_start_s: float
    comparison_stop_s: float
    samples: int
    reference_hr_relative_l2: float
    reference_ht_relative_l2: float
    anomaly_hr_relative_l2: float
    anomaly_ht_relative_l2: float
    perturbation_hr_relative_l2: float
    perturbation_ht_relative_l2: float


def build_paper_adaptive_mesh(
    target_subdivision: int,
    *,
    base_subdivision: int = PAPER_ADAPTIVE_BASE_SUBDIVISION,
    core_radius_deg: float = PAPER_ADAPTIVE_CORE_RADIUS_DEG,
    transition_width_deg: float = PAPER_ADAPTIVE_TRANSITION_WIDTH_DEG,
) -> GeodesicMesh:
    """Build the shared source/oil composite mesh for one convergence level."""

    if target_subdivision <= base_subdivision:
        raise ValueError("target subdivision must exceed the adaptive base")
    regions = (
        SphericalRefinementRegion(
            PAPER_TRANSMITTER_LATITUDE_DEG,
            PAPER_TRANSMITTER_LONGITUDE_DEG,
            core_radius_deg,
            target_subdivision,
            transition_width_deg,
            "transmitter",
        ),
        SphericalRefinementRegion(
            PAPER_OIL_LATITUDE_DEG,
            PAPER_OIL_LONGITUDE_DEG,
            core_radius_deg,
            target_subdivision,
            transition_width_deg,
            "oil-receiver",
        ),
    )
    return build_adaptive_geodesic_mesh(base_subdivision, regions)


def compare_radar_resolution_pairs(
    coarse_reference: RadarTraces,
    coarse_anomaly: RadarTraces,
    fine_reference: RadarTraces,
    fine_anomaly: RadarTraces,
    *,
    coarse_target_subdivision: int,
    fine_target_subdivision: int,
    relative_start_s: float = 0.0,
    relative_stop_s: float = PAPER_FIGURE_7_DURATION_S,
) -> RadarResolutionConvergence:
    """Compare paired-mesh radar fields on the fine run's time samples."""

    if coarse_target_subdivision >= fine_target_subdivision:
        raise ValueError("coarse target subdivision must be below fine target")
    for reference, anomaly in (
        (coarse_reference, coarse_anomaly),
        (fine_reference, fine_anomaly),
    ):
        if reference.case != "reference" or anomaly.case != "anomaly":
            raise ValueError("each resolution must provide reference then anomaly")
        if reference.run_signature != anomaly.run_signature:
            raise ValueError("each resolution pair must share one run signature")
        if reference.source_center_s != anomaly.source_center_s:
            raise ValueError("each resolution pair must share one source center")
        if not np.array_equal(reference.time_s, anomaly.time_s):
            raise ValueError("each resolution pair must share one time grid")
    if coarse_reference.source_center_s != fine_reference.source_center_s:
        raise ValueError("resolution pairs must share one source center")
    if not 0.0 <= relative_start_s < relative_stop_s:
        raise ValueError("comparison window must be positive and ordered")

    source_center = fine_reference.source_center_s
    coarse_time = coarse_reference.time_s - source_center
    fine_time = fine_reference.time_s - source_center
    common_start = max(relative_start_s, float(coarse_time[0]), float(fine_time[0]))
    common_stop = min(relative_stop_s, float(coarse_time[-1]), float(fine_time[-1]))
    selected = (fine_time >= common_start) & (fine_time <= common_stop)
    comparison_time = fine_time[selected]
    if len(comparison_time) < 2:
        raise ValueError("comparison window must contain at least two fine samples")

    def scalar(values: FloatArray) -> FloatArray:
        return np.interp(comparison_time, coarse_time, values)

    def vector(east: FloatArray, north: FloatArray) -> FloatArray:
        return np.column_stack((scalar(east), scalar(north)))

    coarse_fields = {
        "reference_hr": scalar(coarse_reference.hr_a_m),
        "reference_ht": vector(
            coarse_reference.ht_east_a_m, coarse_reference.ht_north_a_m
        ),
        "anomaly_hr": scalar(coarse_anomaly.hr_a_m),
        "anomaly_ht": vector(
            coarse_anomaly.ht_east_a_m, coarse_anomaly.ht_north_a_m
        ),
    }
    fine_fields = {
        "reference_hr": fine_reference.hr_a_m[selected],
        "reference_ht": np.column_stack(
            (
                fine_reference.ht_east_a_m[selected],
                fine_reference.ht_north_a_m[selected],
            )
        ),
        "anomaly_hr": fine_anomaly.hr_a_m[selected],
        "anomaly_ht": np.column_stack(
            (
                fine_anomaly.ht_east_a_m[selected],
                fine_anomaly.ht_north_a_m[selected],
            )
        ),
    }

    def relative_l2(coarse: FloatArray, fine: FloatArray, label: str) -> float:
        denominator = float(np.linalg.norm(fine.ravel()))
        if denominator == 0.0:
            raise ValueError(f"fine {label} field is identically zero")
        return float(np.linalg.norm((coarse - fine).ravel()) / denominator)

    reference_hr = relative_l2(
        coarse_fields["reference_hr"], fine_fields["reference_hr"], "reference Hr"
    )
    reference_ht = relative_l2(
        coarse_fields["reference_ht"], fine_fields["reference_ht"], "reference Ht"
    )
    anomaly_hr = relative_l2(
        coarse_fields["anomaly_hr"], fine_fields["anomaly_hr"], "anomaly Hr"
    )
    anomaly_ht = relative_l2(
        coarse_fields["anomaly_ht"], fine_fields["anomaly_ht"], "anomaly Ht"
    )
    perturbation_hr = relative_l2(
        coarse_fields["anomaly_hr"] - coarse_fields["reference_hr"],
        fine_fields["anomaly_hr"] - fine_fields["reference_hr"],
        "perturbation Hr",
    )
    perturbation_ht = relative_l2(
        coarse_fields["anomaly_ht"] - coarse_fields["reference_ht"],
        fine_fields["anomaly_ht"] - fine_fields["reference_ht"],
        "perturbation Ht",
    )
    return RadarResolutionConvergence(
        coarse_target_subdivision=coarse_target_subdivision,
        fine_target_subdivision=fine_target_subdivision,
        comparison_start_s=float(comparison_time[0]),
        comparison_stop_s=float(comparison_time[-1]),
        samples=len(comparison_time),
        reference_hr_relative_l2=reference_hr,
        reference_ht_relative_l2=reference_ht,
        anomaly_hr_relative_l2=anomaly_hr,
        anomaly_ht_relative_l2=anomaly_ht,
        perturbation_hr_relative_l2=perturbation_hr,
        perturbation_ht_relative_l2=perturbation_ht,
    )


def radar_radial_altitudes_m() -> tuple[float, ...]:
    """Return the 5-km grid with 1.25-km lithosphere surface subgridding."""

    coarse = np.linspace(-100_000.0, 100_000.0, PAPER_NOMINAL_RADIAL_CELLS + 1)
    refined_lithosphere = np.arange(-5_000.0, 0.0, PAPER_SUBGRID_SPACING_M)
    return tuple(np.unique(np.concatenate((coarse, refined_lithosphere))))


def paper_anomalies(
    *,
    include_oil: bool,
    include_shield: bool = True,
    shield_radius_m: float = 2_500_000.0,
    shield_background_resistivity_ohm_m: float = 500.0,
    oil_surface_altitude_m: float = 0.0,
) -> tuple[SphericalAnomaly, ...]:
    """Return the approximate Laurentian Shield and optional oil anomaly."""

    # The paper gives the Shield conductivity but no downloadable boundary.
    # A broad cap centered over Canada includes Clam Lake and most of Canada.
    anomalies: list[SphericalAnomaly] = []
    if include_shield:
        if shield_radius_m <= 0.0:
            raise ValueError("shield_radius_m must be positive")
        if shield_background_resistivity_ohm_m <= 0.0:
            raise ValueError("shield background resistivity must be positive")
        anomalies.append(
            SphericalAnomaly(
                latitude_deg=58.0,
                longitude_deg=-95.0,
                radius_m=shield_radius_m,
                altitude_min_m=-20_000.0,
                altitude_max_m=-1.0,
                conductivity_factor=(
                    2.4e-4 / (1.0 / shield_background_resistivity_ohm_m)
                ),
                maximum_background_conductivity_s_m=0.01,
            )
        )
    if not include_oil:
        return tuple(anomalies)
    half_thickness = 0.5 * PAPER_OIL_THICKNESS_M
    oil = SphericalAnomaly(
        latitude_deg=PAPER_OIL_LATITUDE_DEG,
        longitude_deg=PAPER_OIL_LONGITUDE_DEG,
        radius_m=PAPER_OIL_RADIUS_M,
        altitude_min_m=oil_surface_altitude_m
        - (PAPER_OIL_MEDIAN_DEPTH_M + half_thickness),
        altitude_max_m=oil_surface_altitude_m
        - (PAPER_OIL_MEDIAN_DEPTH_M - half_thickness),
        conductivity_factor=PAPER_OIL_CONDUCTIVITY_FACTOR,
        maximum_background_conductivity_s_m=(
            THESIS_OIL_MAXIMUM_BACKGROUND_CONDUCTIVITY_S_M
        ),
        target_area_m2=PAPER_OIL_AREA_KM2 * 1.0e6,
    )
    anomalies.append(oil)
    return tuple(anomalies)


def create_radar_simulation(
    *,
    include_oil: bool,
    subdivision: int = PAPER_SUBDIVISION,
    material_model: str = "etopo5",
    etopo5_path: str | Path | None = None,
    device: str = "auto",
    dtype: str = "float64",
    compile_step: bool = True,
    compile_chunk_size: int = 8,
    source_center_s: float = PAPER_SOURCE_CENTER_S,
    courant_factor: float = 0.4,
    source_edge_assignment: str = "projected",
    tangential_interface_mode: str = "point",
    tangential_material_support: str = "point",
    source_altitude_m: float | None = None,
    source_azimuths_deg: tuple[float, ...] = (0.0, 90.0),
    include_shield: bool = True,
    shield_radius_m: float = 2_500_000.0,
    mesh_orientation: str = "polar",
    mesh_optimization_steps: int = 0,
    mesh: GeodesicMesh | None = None,
    geometry_mode: str = "full-spherical",
    vertical_reference: str = "terrain",
    horizontal_anomaly_mode: str = "conservative-nearest",
    deep_lithosphere_resistivity_ohm_m: float = (
        REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M
    ),
    upper_crust_resistivity_ohm_m: float = 500.0,
    asthenosphere_resistivity_ohm_m: float = 200.0,
    lithosphere_profile: str = "legacy",
    ionosphere_model: str = "daytime",
    subsolar_latitude_deg: float = 0.0,
    subsolar_longitude_deg: float = THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
    nighttime_ionosphere_reference_height_m: float = (
        THESIS_NIGHTTIME_IONOSPHERE_REFERENCE_HEIGHT_M
    ),
    nighttime_ionosphere_scale_height_m: float = (
        THESIS_NIGHTTIME_IONOSPHERE_SCALE_HEIGHT_M
    ),
) -> GeodesicFDTD:
    """Create one reference or oil-anomaly model for Figure 7."""

    if vertical_reference not in {"sea-level", "terrain"}:
        raise ValueError("vertical_reference must be 'sea-level' or 'terrain'")
    material_arguments: dict[str, Any] = {
        "tangential_interface_mode": tangential_interface_mode,
    }
    if lithosphere_profile == "legacy":
        material_class = SimpsonTaflove2004Material
        material_arguments.update(
            upper_crust_resistivity_ohm_m=upper_crust_resistivity_ohm_m,
            asthenosphere_resistivity_ohm_m=asthenosphere_resistivity_ohm_m,
            deep_rock_resistivity_ohm_m=deep_lithosphere_resistivity_ohm_m,
        )
        shield_background_resistivity_ohm_m = upper_crust_resistivity_ohm_m
    elif lithosphere_profile == "figure-15":
        material_class = HermanceFigure15Material
        shield_background_resistivity_ohm_m = (
            THESIS_FIGURE_15_CONTINENTAL_RESISTIVITY_LIMIT_OHM_M
        )
    else:
        raise ValueError("lithosphere_profile must be 'legacy' or 'figure-15'")
    if ionosphere_model == "day-night":
        shared_solar_geometry = {
            "subsolar_latitude_deg": subsolar_latitude_deg,
            "subsolar_longitude_deg": subsolar_longitude_deg,
        }
        material_arguments.update(
            ionosphere_reference_height_sampler=DayNightHemisphereProfile(
                PAPER_DAYTIME_IONOSPHERE_REFERENCE_HEIGHT_M,
                nighttime_ionosphere_reference_height_m,
                **shared_solar_geometry,
            ),
            ionosphere_scale_height_sampler=DayNightHemisphereProfile(
                PAPER_DAYTIME_IONOSPHERE_SCALE_HEIGHT_M,
                nighttime_ionosphere_scale_height_m,
                **shared_solar_geometry,
            ),
        )
    elif ionosphere_model != "daytime":
        raise ValueError("ionosphere_model must be 'daytime' or 'day-night'")
    relief: ETOPO5Relief | None = None
    if material_model == "etopo5":
        if etopo5_path is None:
            raise ValueError("etopo5_path is required for the ETOPO5 material")
        relief = ETOPO5Relief.from_file(etopo5_path)
        material_arguments["surface_elevation_sampler"] = relief
    elif material_model == "natural-earth":
        material_arguments["land_classifier"] = natural_earth_land_classifier()
    else:
        raise ValueError("material_model must be 'etopo5' or 'natural-earth'")
    source_surface_altitude_m = 0.0
    receiver_surface_altitude_m = 0.0
    if vertical_reference == "terrain" and relief is not None:
        surface_directions = np.stack(
            (
                geographic_direction(
                    PAPER_TRANSMITTER_LATITUDE_DEG,
                    PAPER_TRANSMITTER_LONGITUDE_DEG,
                ),
                geographic_direction(PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG),
            )
        )
        source_surface_altitude_m, receiver_surface_altitude_m = (
            float(value) for value in relief(surface_directions)
        )
    anomalies = paper_anomalies(
        include_oil=include_oil,
        include_shield=include_shield,
        shield_radius_m=shield_radius_m,
        shield_background_resistivity_ohm_m=(
            shield_background_resistivity_ohm_m
        ),
        oil_surface_altitude_m=receiver_surface_altitude_m,
    )
    material_arguments["anomalies"] = anomalies
    material = material_class(**material_arguments)
    effective_source_altitude_m = (
        source_surface_altitude_m
        if source_altitude_m is None
        else source_altitude_m
    )
    source = TangentialGaussianCurrent(
        latitude_deg=PAPER_TRANSMITTER_LATITUDE_DEG,
        longitude_deg=PAPER_TRANSMITTER_LONGITUDE_DEG,
        altitude_m=effective_source_altitude_m,
        peak_current_a=PAPER_TRANSMITTER_CURRENT_A,
        center_time_s=source_center_s,
        one_over_e_half_width_s=PAPER_ENVELOPE_ONE_OVER_E_HALF_WIDTH_S,
        carrier_frequency_hz=PAPER_CARRIER_FREQUENCY_HZ,
        azimuths_deg=source_azimuths_deg,
        line_lengths_m=tuple(
            PAPER_TRANSMITTER_LINE_LENGTH_M for _ in source_azimuths_deg
        ),
        edge_assignment=source_edge_assignment,
    )
    altitudes = radar_radial_altitudes_m()
    simulation = GeodesicFDTD(
        config=SimulationConfig(
            subdivision=subdivision,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            courant_factor=courant_factor,
            radial_altitudes_m=altitudes,
            radial_grid_policy="allow-abrupt",
            mesh_orientation=mesh_orientation,
            mesh_optimization_steps=mesh_optimization_steps,
            tangential_material_support=tangential_material_support,
            horizontal_anomaly_mode=horizontal_anomaly_mode,
            loss_integration="trapezoidal",
            geometry_mode=geometry_mode,
        ),
        material=material,
        source=source,
        device=device,
        dtype=dtype,
        compile_step=compile_step,
        compile_chunk_size=compile_chunk_size,
        mesh=mesh,
    )
    simulation.radar_receiver_altitude_m = receiver_surface_altitude_m
    simulation.radar_vertical_reference = vertical_reference
    return simulation


def _surface_h_distributions(
    simulation: GeodesicFDTD,
    *,
    receiver_support: str = "local-linear",
) -> tuple[NDArray[np.int64], ...]:
    if receiver_support not in {"face", "local-linear"}:
        raise ValueError("receiver_support must be 'face' or 'local-linear'")
    receiver_altitude_m = float(
        getattr(simulation, "radar_receiver_altitude_m", 0.0)
    )
    face = geographic_face_index(
        simulation, PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG
    )
    hr_layers, hr_weights = radial_linear_distribution(
        simulation.radial_midpoint_altitudes_m, receiver_altitude_m
    )
    if receiver_support == "face":
        support_faces = np.asarray((face,), dtype=np.int64)
        horizontal_weights = np.asarray((1.0,))
    else:
        edges = simulation.mesh.face_edges[face]
        edge_faces = np.column_stack(
            (
                simulation.mesh.edge_left_faces[edges],
                simulation.mesh.edge_right_faces[edges],
            )
        )
        neighbors = np.asarray(
            sorted(set(edge_faces.ravel()) - {face}), dtype=np.int64
        )
        support_faces = np.concatenate((np.asarray((face,)), neighbors))
        if len(support_faces) != 4:
            raise RuntimeError("face receiver must have three neighbors")
        radial = geographic_direction(
            PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG
        )
        east, north = geographic_tangent_basis(
            PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG
        )
        centers = simulation.mesh.face_centers[support_faces]
        coordinates = np.column_stack((centers @ east, centers @ north))
        constraints = np.vstack(
            (np.ones(len(support_faces)), coordinates.T)
        )
        target = np.asarray((1.0, 0.0, 0.0))
        horizontal_weights = constraints.T @ np.linalg.solve(
            constraints @ constraints.T, target
        )
    hr_faces = np.repeat(support_faces, len(hr_layers))[None, :]
    hr_layer_indices = np.tile(hr_layers, len(support_faces))[None, :]
    hr_sample_weights = (
        np.repeat(horizontal_weights, len(hr_layers))
        * np.tile(hr_weights, len(support_faces))
    )[None, :]

    edges = simulation.mesh.face_edges[face]
    left = simulation.mesh.face_centers[simulation.mesh.edge_left_faces[edges]]
    right = simulation.mesh.face_centers[simulation.mesh.edge_right_faces[edges]]
    dual_directions = left - right
    radial = geographic_direction(PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG)
    dual_directions -= (dual_directions @ radial)[:, None] * radial[None, :]
    dual_directions /= np.linalg.norm(dual_directions, axis=1, keepdims=True)
    east, north = geographic_tangent_basis(
        PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG
    )
    samples = np.column_stack((dual_directions @ east, dual_directions @ north))
    reconstruction = samples @ np.linalg.inv(samples.T @ samples)
    ht_radial_layers, ht_radial_weights = radial_linear_distribution(
        simulation.altitudes_m, receiver_altitude_m
    )
    ht_support_edges = np.repeat(edges, len(ht_radial_layers))
    ht_support_layers = np.tile(ht_radial_layers, len(edges))
    ht_edges = np.stack((ht_support_edges, ht_support_edges))
    ht_layers = np.stack((ht_support_layers, ht_support_layers))
    ht_weights = np.stack(
        tuple(
            np.repeat(reconstruction[:, component], len(ht_radial_layers))
            * np.tile(ht_radial_weights, len(edges))
            for component in range(2)
        )
    )
    return (
        hr_faces,
        hr_layer_indices,
        hr_sample_weights,
        ht_edges,
        ht_layers,
        ht_weights,
    )


def record_radar_traces(
    simulation: GeodesicFDTD,
    *,
    steps: int,
    case: str,
    synchronize_every: int = 128,
    receiver_support: str = "local-linear",
    sample_every: int = 1,
) -> RadarTraces:
    """Record interpolated surface ``Hr`` and east/north ``Htan`` traces."""

    if simulation.steps != 0:
        raise ValueError("radar recording requires a fresh simulation")
    if case not in {"reference", "anomaly"}:
        raise ValueError("radar case must be 'reference' or 'anomaly'")
    run_signature = _radar_run_signature(
        simulation,
        case=case,
        receiver_support=receiver_support,
    )
    distributions = _surface_h_distributions(
        simulation, receiver_support=receiver_support
    )
    hr, ht = simulation.record_h_observations(
        *distributions,
        steps,
        synchronize_every=synchronize_every,
        sample_every=sample_every,
    )
    hr = simulation.to_numpy(hr)
    ht = simulation.to_numpy(ht)
    sample_steps = np.concatenate(
        (
            np.arange(0, steps + 1, sample_every, dtype=np.int64),
            np.asarray((steps,), dtype=np.int64),
        )
    )
    sample_steps = np.unique(sample_steps)
    time_s = (sample_steps.astype(np.float64) - 0.5) * simulation.time_step_s
    source_center = (
        simulation.source.center_time_s
        if simulation.source is not None
        else PAPER_SOURCE_CENTER_S
    )
    assert source_center is not None
    return RadarTraces(
        time_s=time_s,
        hr_a_m=hr[:, 0].astype(np.float64, copy=False),
        ht_east_a_m=ht[:, 0].astype(np.float64, copy=False),
        ht_north_a_m=ht[:, 1].astype(np.float64, copy=False),
        source_center_s=source_center,
        case=case,
        run_signature=run_signature,
    )


def save_radar_traces(traces: RadarTraces, path: str | Path) -> Path:
    """Save a compact, self-describing radar trace archive."""

    return save_npz_atomic(
        path,
        format_version=np.asarray(2),
        time_s=traces.time_s,
        hr_a_m=traces.hr_a_m,
        ht_east_a_m=traces.ht_east_a_m,
        ht_north_a_m=traces.ht_north_a_m,
        source_center_s=np.asarray(traces.source_center_s),
        case=np.asarray(traces.case),
        run_signature=np.asarray(traces.run_signature),
    )


def load_radar_traces(path: str | Path) -> RadarTraces:
    """Load a radar trace archive written by :func:`save_radar_traces`."""

    with np.load(path) as values:
        if int(values["format_version"]) != 2:
            raise ValueError("unsupported radar trace format")
        return RadarTraces(
            time_s=values["time_s"].astype(np.float64),
            hr_a_m=values["hr_a_m"].astype(np.float64),
            ht_east_a_m=values["ht_east_a_m"].astype(np.float64),
            ht_north_a_m=values["ht_north_a_m"].astype(np.float64),
            source_center_s=float(values["source_center_s"]),
            case=str(values["case"]),
            run_signature=str(values["run_signature"]),
        )


def compute_radar_perturbation(
    reference: RadarTraces,
    anomaly: RadarTraces,
    *,
    relative_start_s: float = 0.0,
    relative_stop_s: float = PAPER_FIGURE_7_DURATION_S,
    denominator_floor_fraction: float = 1.0e-6,
    normalization: str = "pointwise",
    ht_definition: str = "vector-difference",
) -> RadarPerturbation:
    """Compute perturbations for an explicit tangential-field definition."""

    if reference.case != "reference" or anomaly.case != "anomaly":
        raise ValueError("radar traces must be ordered as reference then anomaly")
    if reference.run_signature != anomaly.run_signature:
        raise ValueError("reference and anomaly run signatures do not match")
    if reference.source_center_s != anomaly.source_center_s:
        raise ValueError("reference and anomaly source centers do not match")
    if reference.time_s.shape != anomaly.time_s.shape or not np.allclose(
        reference.time_s, anomaly.time_s, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError("reference and anomaly traces must use the same time grid")
    relative_time = reference.time_s - reference.source_center_s
    selected = (relative_time >= relative_start_s) & (
        relative_time <= relative_stop_s
    )
    if not np.any(selected):
        raise ValueError("requested Figure 7 window is absent from the traces")
    if normalization not in {"pointwise", "peak"}:
        raise ValueError("normalization must be 'pointwise' or 'peak'")

    reference_ht_vector = np.column_stack(
        (reference.ht_east_a_m[selected], reference.ht_north_a_m[selected])
    )
    anomaly_ht_vector = np.column_stack(
        (anomaly.ht_east_a_m[selected], anomaly.ht_north_a_m[selected])
    )
    reference_ht, _, difference_ht, projection = _tangential_response(
        reference_ht_vector, anomaly_ht_vector, definition=ht_definition
    )
    reference_hr = reference.hr_a_m[selected]
    anomaly_hr = anomaly.hr_a_m[selected]

    def relative_db(
        base: FloatArray, difference: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        peak = float(np.max(np.abs(base)))
        if peak == 0.0:
            raise ValueError("reference magnetic field is identically zero")
        if normalization == "pointwise":
            denominator = np.abs(base)
            valid = denominator >= denominator_floor_fraction * peak
        else:
            denominator = np.full_like(base, peak)
            valid = np.ones_like(base, dtype=np.bool_)
        ratio = np.full_like(base, np.nan)
        ratio[valid] = difference[valid] / denominator[valid]
        ratio[valid] = np.maximum(ratio[valid], np.finfo(np.float64).tiny)
        return 20.0 * np.log10(ratio), valid

    delta_ht_db, valid_ht = relative_db(reference_ht, difference_ht)
    delta_hr_db, valid_hr = relative_db(
        reference_hr, np.abs(anomaly_hr - reference_hr)
    )
    return RadarPerturbation(
        time_s=relative_time[selected],
        delta_ht_db=delta_ht_db,
        delta_hr_db=delta_hr_db,
        valid_ht=valid_ht,
        valid_hr=valid_hr,
        ht_projection_east_north=projection,
        normalization=normalization,
        ht_definition=ht_definition,
    )


def _tangential_response(
    reference: FloatArray,
    anomaly: FloatArray,
    *,
    definition: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Reduce two-component tangential fields using one declared convention."""

    if definition == "principal-axis":
        _, _, principal_axes = np.linalg.svd(reference, full_matrices=False)
        projection = principal_axes[0]
        base = reference @ projection
        changed = anomaly @ projection
        difference = np.abs(changed - base)
    elif definition in {"east", "north"}:
        component = 0 if definition == "east" else 1
        projection = np.eye(2, dtype=np.float64)[component]
        base = reference[:, component]
        changed = anomaly[:, component]
        difference = np.abs(changed - base)
    elif definition in {"magnitude", "vector-difference"}:
        projection = np.full(2, np.nan, dtype=np.float64)
        base = np.linalg.norm(reference, axis=1)
        changed = np.linalg.norm(anomaly, axis=1)
        if definition == "magnitude":
            difference = np.abs(changed - base)
        else:
            difference = np.linalg.norm(anomaly - reference, axis=1)
    else:
        raise ValueError(
            "ht_definition must be 'principal-axis', 'east', 'north', "
            "'magnitude', or 'vector-difference'"
        )
    return base, changed, difference, projection


def _radar_run_signature(
    simulation: GeodesicFDTD,
    *,
    case: str,
    receiver_support: str,
) -> str:
    """Return a canonical signature shared by a valid Figure 7 run pair."""

    material_fields = {
        field.name: getattr(simulation.material, field.name)
        for field in fields(simulation.material)
        if field.name != "anomalies"
    }
    anomalies = list(getattr(simulation.material, "anomalies", ()))
    oil_indices = [
        index for index, anomaly in enumerate(anomalies) if _is_paper_oil(anomaly)
    ]
    expected_oil_count = 0 if case == "reference" else 1
    if len(oil_indices) != expected_oil_count:
        raise ValueError(
            f"{case} radar model must contain {expected_oil_count} paper oil anomaly"
        )
    shared_anomalies = [
        anomaly for index, anomaly in enumerate(anomalies) if index not in oil_indices
    ]
    payload = {
        "format_version": 1,
        "git_revision": _git_revision(),
        "mesh_vertices_sha256": _array_sha256(simulation.mesh.vertices),
        "mesh_faces_sha256": hashlib.sha256(
            np.ascontiguousarray(simulation.mesh.faces, dtype="<i8").tobytes()
        ).hexdigest(),
        "config": _signature_value(simulation.config),
        "time_step_s": simulation.time_step_s,
        "material_class": _qualified_name(simulation.material),
        "material": _signature_value(material_fields),
        "shared_anomalies": _signature_value(shared_anomalies),
        "source_class": (
            _qualified_name(simulation.source) if simulation.source is not None else None
        ),
        "source": _signature_value(simulation.source),
        "receiver_support": receiver_support,
        "receiver_altitude_m": float(
            getattr(simulation, "radar_receiver_altitude_m", 0.0)
        ),
        "vertical_reference": str(
            getattr(simulation, "radar_vertical_reference", "sea-level")
        ),
        "runtime": simulation.runtime,
        "dtype": simulation.dtype_name,
        "compiled": simulation.compiled,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _is_paper_oil(anomaly: Any) -> bool:
    if not isinstance(anomaly, SphericalAnomaly):
        return False
    expected = (
        PAPER_OIL_LATITUDE_DEG,
        PAPER_OIL_LONGITUDE_DEG,
        PAPER_OIL_RADIUS_M,
        PAPER_OIL_THICKNESS_M,
        PAPER_OIL_CONDUCTIVITY_FACTOR,
    )
    actual = (
        anomaly.latitude_deg,
        anomaly.longitude_deg,
        anomaly.radius_m,
        anomaly.altitude_max_m - anomaly.altitude_min_m,
        anomaly.conductivity_factor,
    )
    return bool(np.allclose(actual, expected, rtol=0.0, atol=1.0e-12))


def _signature_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, ETOPO5Relief):
        return {
            "class": _qualified_name(value),
            "path": str(value.path.expanduser().resolve()),
            "sha256": ETOPO5_SHA256,
        }
    if is_dataclass(value):
        return {
            field.name: _signature_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _signature_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_signature_value(item) for item in value]
    if callable(value):
        return {"callable": _qualified_name(value)}
    raise TypeError(f"unsupported radar signature value: {type(value).__name__}")


def _qualified_name(value: Any) -> str:
    target = value if callable(value) and not is_dataclass(value) else type(value)
    return f"{target.__module__}.{target.__qualname__}"


def _array_sha256(values: NDArray[np.generic]) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _git_revision() -> str:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision}-dirty" if dirty else revision


def radar_metrics(curves: RadarPerturbation) -> dict[str, float]:
    """Summarize Figure 7 away from reference zero-crossing singularities."""

    ht = curves.delta_ht_db[curves.valid_ht]
    hr = curves.delta_hr_db[curves.valid_hr]
    common = curves.valid_ht & curves.valid_hr
    return {
        "delta_ht_median_db": float(np.median(ht)),
        "delta_ht_fraction_below_minus_25_db": float(np.mean(ht < -25.0)),
        "delta_hr_median_db": float(np.median(hr)),
        "delta_hr_95th_percentile_db": float(np.percentile(hr, 95.0)),
        "delta_hr_maximum_db": float(np.max(hr)),
        "median_hr_over_ht_advantage_db": float(
            np.median(curves.delta_hr_db[common] - curves.delta_ht_db[common])
        ),
    }


def radar_field_metrics(
    reference: RadarTraces,
    anomaly: RadarTraces,
    curves: RadarPerturbation,
) -> dict[str, float]:
    """Report absolute peaks and the paper-body peak normalization."""

    relative_time = reference.time_s - reference.source_center_s
    selected = (relative_time >= curves.time_s[0]) & (
        relative_time <= curves.time_s[-1]
    )
    reference_ht_vector = np.column_stack(
        (reference.ht_east_a_m[selected], reference.ht_north_a_m[selected])
    )
    anomaly_ht_vector = np.column_stack(
        (anomaly.ht_east_a_m[selected], anomaly.ht_north_a_m[selected])
    )
    reference_ht, anomaly_ht, difference_ht, _ = _tangential_response(
        reference_ht_vector,
        anomaly_ht_vector,
        definition=curves.ht_definition,
    )
    fields = {
        "ht": (reference_ht, anomaly_ht, difference_ht),
        "hr": (
            reference.hr_a_m[selected],
            anomaly.hr_a_m[selected],
            np.abs(anomaly.hr_a_m[selected] - reference.hr_a_m[selected]),
        ),
    }
    metrics: dict[str, float] = {}
    for name, (base, changed, difference) in fields.items():
        reference_peak = float(np.max(np.abs(base)))
        anomaly_peak = float(np.max(np.abs(changed)))
        difference_peak = float(np.max(difference))
        metrics.update(
            {
                f"{name}_reference_peak_a_m": reference_peak,
                f"{name}_anomaly_peak_a_m": anomaly_peak,
                f"delta_{name}_peak_a_m": difference_peak,
                f"delta_{name}_peak_normalized_db": float(
                    20.0 * np.log10(difference_peak / reference_peak)
                ),
            }
        )
    return metrics


def normalized_figure_5_traces(
    traces: ValidationTraces,
) -> dict[str, FloatArray]:
    """Return all four Figure 5 records with one common normalization."""

    values = {label: -traces.trace(label) for label in ("A", "A′", "B", "B′")}
    scale = float(max(np.max(np.abs(trace)) for trace in values.values()))
    if scale == 0.0:
        raise ValueError("Figure 5 traces are identically zero")
    return {label: trace / scale for label, trace in values.items()}


def render_figure_5(traces: ValidationTraces, path: str | Path) -> Path:
    """Render the four normalized geodesic-grid records of Figure 5."""

    import matplotlib.pyplot as plt

    values = normalized_figure_5_traces(traces)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    styles = {
        "A": {"color": "black", "linestyle": "-"},
        "A′": {"color": "0.45", "linestyle": "-"},
        "B": {"color": "black", "linestyle": ":", "linewidth": 2.0},
        "B′": {"color": "0.45", "linestyle": ":", "linewidth": 2.0},
    }
    for label, trace in values.items():
        axis.plot(traces.time_s, trace, label=f"Point {label}", **styles[label])
    axis.set(
        xlim=(0.0, 0.12),
        ylim=(-0.2, 1.0),
    )
    axis.set_xticks(np.arange(0.0, 0.121, 0.02))
    axis.set_yticks(np.arange(-0.1, 1.01, 0.1))
    axis.set_xlabel("Time (seconds)", fontsize=22)
    axis.set_ylabel("Normalized radial electric field", fontsize=22)
    axis.tick_params(axis="both", which="major", labelsize=20)
    axis.legend(frameon=False, ncol=2, fontsize=20)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def render_figure_6(curves: AttenuationCurves, path: str | Path) -> Path:
    """Render the Figure 6 attenuation comparison."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    simulated = (
        (curves.frequency_hz >= curves.valid_frequency_hz[0])
        & (curves.frequency_hz <= curves.valid_frequency_hz[1])
    )
    guide_frequency = np.geomspace(5.0, 2_000.0, 512)
    x_ticks = (5, 10, 20, 50, 100, 200, 500, 1_000, 2_000)
    y_ticks = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30)
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.loglog(
        curves.frequency_hz[simulated],
        curves.path_ab_db_per_mm[simulated],
        color="black",
        label="East of source",
    )
    axis.loglog(
        curves.frequency_hz[simulated],
        curves.path_apbp_db_per_mm[simulated],
        color="0.55",
        label="West of source",
    )
    axis.loglog(
        guide_frequency,
        bannister_figure_8_guide(guide_frequency),
        color="black",
        linestyle=":",
        label="Previous theoretical results",
    )
    axis.set(
        xlim=(5.0, 2_000.0),
        ylim=(0.1, 30.0),
    )
    axis.set_xticks(x_ticks, labels=[f"{value:g}" for value in x_ticks])
    axis.set_yticks(y_ticks, labels=[f"{value:g}" for value in y_ticks])
    axis.set_xlabel("Frequency (Hz)", fontsize=22)
    axis.set_ylabel("Attenuation rate (dB/Mm)", fontsize=22)
    axis.tick_params(axis="both", which="major", labelsize=20)
    axis.legend(frameon=False, fontsize=20)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def render_figure_7(curves: RadarPerturbation, path: str | Path) -> Path:
    """Render the normalized surface magnetic perturbations of Figure 7."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(
        curves.time_s,
        curves.delta_ht_db,
        color="black",
        linestyle="--",
        label="(a) ΔHtan",
    )
    axis.plot(
        curves.time_s,
        curves.delta_hr_db,
        color="black",
        label="(b) ΔHr",
    )
    axis.set(
        xlim=(0.0, PAPER_FIGURE_7_DURATION_S),
        ylim=(-100.0, 30.0),
    )
    axis.set_xticks(np.arange(0.0, 0.081, 0.01))
    axis.set_yticks(np.arange(-100.0, 21.0, 20.0))
    axis.set_xlabel("Time (seconds)", fontsize=20)
    axis.set_ylabel("Surface magnetic field perturbation (dB)", fontsize=20)
    axis.tick_params(axis="both", which="major", labelsize=18)
    axis.legend(frameon=False, fontsize=18)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output

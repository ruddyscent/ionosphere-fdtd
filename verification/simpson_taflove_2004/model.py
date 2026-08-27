"""Reproduce the validation study in Simpson and Taflove (2004), Figs. 7–8."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ionosphere_fdtd.constants import C_0, EARTH_RADIUS_M
from ionosphere_fdtd.materials import EarthIonosphereMaterial
from ionosphere_fdtd.mesh import GeodesicMesh, build_geodesic_mesh
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent, geographic_distribution

from ..physics_diagnostics.model import (
    HorizontalRegion,
    record_er_observations_with_diagnostics,
)
from .materials import ETOPO5Relief, SimpsonTaflove2004Material

PAPER_TIME_STEP_S = 3.0e-6
PAPER_RADIAL_CELLS = 40
PAPER_SOURCE_LATITUDE_DEG = 0.0
PAPER_SOURCE_LONGITUDE_DEG = -47.0
PAPER_SOURCE_ALTITUDE_M = 2_500.0
PAPER_SOURCE_PEAK_CURRENT_A = 1.0
PAPER_SOURCE_FULL_WIDTH_STEPS = 480
PAPER_SOURCE_CENTER_STEPS = 960
PAPER_TRACE_STEPS = 35_000
PAPER_DFT_TRUNCATIONS = {"A": 22_849, "B": 24_165, "A′": 22_737, "B′": 25_023}
PAPER_DFT_SIZE = 32_768
REPRESENTATIVE_IONOSPHERE_REFERENCE_HEIGHT_M = 70_000.0
REPRESENTATIVE_IONOSPHERE_SCALE_HEIGHT_M = 1_000.0 / 0.3
REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M = 500.0
PAPER_VALID_FREQUENCY_HZ = (50.0, 500.0)
PAPER_PATH_AB_TOLERANCE_DB_PER_MM = 0.5
PAPER_PATH_APBP_TOLERANCE_DB_PER_MM = 1.0
PAPER_MINIMUM_SIMULATION_STEPS = max(PAPER_DFT_TRUNCATIONS.values()) - 1
BANNISTER_REFERENCE_HEIGHT_KM = 70.0
BANNISTER_SCALE_HEIGHT_KM = 1.0 / 0.3

_paper_frequency_resolution_hz = 1.0 / (PAPER_DFT_SIZE * PAPER_TIME_STEP_S)
PAPER_EVALUATION_FREQUENCIES_HZ = tuple(
    np.arange(
        np.ceil(PAPER_VALID_FREQUENCY_HZ[0] / _paper_frequency_resolution_hz),
        np.floor(PAPER_VALID_FREQUENCY_HZ[1] / _paper_frequency_resolution_hz) + 1,
    )
    * _paper_frequency_resolution_hz
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PaperReceiver:
    """One equatorial observation point from Figure 7."""

    label: str
    longitude_deg: float
    fraction_to_antipode: float
    direction: str
    altitude_m: float = 0.0


PAPER_RECEIVERS = (
    PaperReceiver("A", -2.0, 0.25, "east"),
    PaperReceiver("A′", -92.0, 0.25, "west"),
    PaperReceiver("B", 43.0, 0.50, "east"),
    PaperReceiver("B′", -137.0, 0.50, "west"),
)


def equatorial_path_diagnostic_regions(
    simulation: GeodesicFDTD,
    *,
    corridor_half_width_deg: float = 10.0,
    source_exclusion_deg: float = 5.0,
) -> dict[str, HorizontalRegion]:
    """Return equal-width east/west masks for the two Figure 7 paths."""

    if not 0.0 < corridor_half_width_deg <= 90.0:
        raise ValueError("corridor_half_width_deg must be in (0, 90]")
    if not 0.0 <= source_exclusion_deg < 90.0:
        raise ValueError("source_exclusion_deg must be in [0, 90)")

    def masks(points: FloatArray) -> tuple[FloatArray, FloatArray]:
        longitude = np.mod(
            np.rad2deg(np.arctan2(points[:, 1], points[:, 0])), 360.0
        )
        latitude = np.rad2deg(
            np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))
        )
        source_longitude = np.mod(PAPER_SOURCE_LONGITUDE_DEG, 360.0)
        east_distance = np.mod(longitude - source_longitude, 360.0)
        west_distance = np.mod(source_longitude - longitude, 360.0)
        latitude_mask = np.abs(latitude) <= corridor_half_width_deg
        east = (
            latitude_mask
            & (east_distance >= source_exclusion_deg)
            & (east_distance <= 90.0)
        )
        west = (
            latitude_mask
            & (west_distance >= source_exclusion_deg)
            & (west_distance <= 90.0)
        )
        return east.astype(np.float64), west.astype(np.float64)

    vertex_east, vertex_west = masks(simulation.mesh.vertices)
    edge_east, edge_west = masks(simulation.mesh.edge_midpoints())
    face_east, face_west = masks(simulation.mesh.face_centers)
    return {
        "east": HorizontalRegion(vertex_east, edge_east, face_east),
        "west": HorizontalRegion(vertex_west, edge_west, face_west),
    }


@dataclass(frozen=True, slots=True)
class ValidationTraces:
    """Radial electric fields at the four Figure 7 receivers."""

    time_steps: NDArray[np.int64]
    time_s: FloatArray
    er_v_m: FloatArray
    labels: tuple[str, ...]

    def trace(self, label: str) -> FloatArray:
        return self.er_v_m[:, self.labels.index(label)]


@dataclass(frozen=True, slots=True)
class AttenuationCurves:
    """Frequency-domain attenuation for the east and west 45° paths."""

    frequency_hz: FloatArray
    path_ab_db_per_mm: FloatArray
    path_apbp_db_per_mm: FloatArray
    benchmark_db_per_mm: FloatArray
    dft_truncations: Mapping[str, int]
    valid_frequency_hz: tuple[float, float] = PAPER_VALID_FREQUENCY_HZ


@dataclass(frozen=True, slots=True)
class PhaseVelocityCurves:
    """Phase velocity over the extra 45-degree receiver path."""

    frequency_hz: FloatArray
    path_ab_fraction_c: FloatArray
    path_apbp_fraction_c: FloatArray
    benchmark_fraction_c: FloatArray
    dft_truncations: Mapping[str, int]


def natural_earth_land_classifier() -> Callable[[FloatArray], NDArray[np.bool_]]:
    """Build a vectorized land classifier from Natural Earth's 110-m polygons."""

    geometry, contains_xy = _natural_earth_geometry()

    def classify(directions: FloatArray) -> NDArray[np.bool_]:
        longitude = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
        latitude = np.rad2deg(
            np.arctan2(
                directions[:, 2],
                np.hypot(directions[:, 0], directions[:, 1]),
            )
        )
        return np.asarray(contains_xy(geometry, longitude, latitude), dtype=np.bool_)

    return classify


@lru_cache(maxsize=1)
def _natural_earth_geometry() -> tuple[Any, Any]:
    try:
        from cartopy.io import shapereader
        from shapely import contains_xy
        from shapely.ops import unary_union
    except ImportError as error:
        raise ImportError(
            "Natural Earth materials require: uv sync --extra visualization"
        ) from error
    path = shapereader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    geometry = unary_union(tuple(shapereader.Reader(path).geometries()))
    return geometry, contains_xy


def create_validation_simulation(
    *,
    subdivision: int = 7,
    radial_cells: int = PAPER_RADIAL_CELLS,
    time_step_s: float = PAPER_TIME_STEP_S,
    material_model: str = "natural-earth",
    device: str = "auto",
    dtype: str = "float32",
    compile_step: bool = True,
    torch_threads: int | None = None,
    mesh_orientation: str = "polar",
    mesh_optimization_steps: int = 0,
    ionosphere_reference_height_m: float = (
        REPRESENTATIVE_IONOSPHERE_REFERENCE_HEIGHT_M
    ),
    ionosphere_scale_height_m: float = REPRESENTATIVE_IONOSPHERE_SCALE_HEIGHT_M,
    etopo5_path: str | Path | None = None,
    radial_material_support: str = "point",
    tangential_interface_mode: str = "point",
    tangential_material_support: str = "point",
    minimum_ocean_depth_m: float = 0.0,
    deep_lithosphere_resistivity_ohm_m: float = (
        REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M
    ),
    mesh: GeodesicMesh | None = None,
    compress_uniform_material_coefficients: bool = False,
) -> GeodesicFDTD:
    """Create the paper's 200-km radial domain, pulse, and 3-µs time step."""

    if material_model == "natural-earth":
        material: Any = SimpsonTaflove2004Material(
            land_classifier=natural_earth_land_classifier(),
            ionosphere_reference_height_m=ionosphere_reference_height_m,
            ionosphere_scale_height_m=ionosphere_scale_height_m,
            tangential_interface_mode=tangential_interface_mode,
            minimum_ocean_depth_m=minimum_ocean_depth_m,
            deep_rock_resistivity_ohm_m=deep_lithosphere_resistivity_ohm_m,
        )
    elif material_model == "etopo5":
        if etopo5_path is None:
            raise ValueError("etopo5_path is required for the ETOPO5 material")
        material = SimpsonTaflove2004Material(
            surface_elevation_sampler=ETOPO5Relief.from_file(etopo5_path),
            ionosphere_reference_height_m=ionosphere_reference_height_m,
            ionosphere_scale_height_m=ionosphere_scale_height_m,
            tangential_interface_mode=tangential_interface_mode,
            minimum_ocean_depth_m=minimum_ocean_depth_m,
            deep_rock_resistivity_ohm_m=deep_lithosphere_resistivity_ohm_m,
        )
    elif material_model == "uniform":
        material = EarthIonosphereMaterial(
            ionosphere_reference_height_m=ionosphere_reference_height_m,
            ionosphere_scale_height_m=ionosphere_scale_height_m,
        )
    else:
        raise ValueError(
            "material_model must be 'natural-earth', 'etopo5', or 'uniform'"
        )
    source = GaussianCurrent(
        latitude_deg=PAPER_SOURCE_LATITUDE_DEG,
        longitude_deg=PAPER_SOURCE_LONGITUDE_DEG,
        altitude_m=PAPER_SOURCE_ALTITUDE_M,
        peak_current_a=PAPER_SOURCE_PEAK_CURRENT_A,
        vertical_element_length_m=5_000.0,
        center_time_s=PAPER_SOURCE_CENTER_STEPS * PAPER_TIME_STEP_S,
        one_over_e_half_width_s=(
            0.5 * PAPER_SOURCE_FULL_WIDTH_STEPS * PAPER_TIME_STEP_S
        ),
    )
    return GeodesicFDTD(
        config=SimulationConfig(
            subdivision=subdivision,
            radial_cells=radial_cells,
            minimum_altitude_m=-100_000.0,
            maximum_altitude_m=100_000.0,
            courant_factor=0.4,
            time_step_s=time_step_s,
            mesh_orientation=mesh_orientation,
            mesh_optimization_steps=mesh_optimization_steps,
            radial_material_support=radial_material_support,
            tangential_material_support=tangential_material_support,
            loss_integration="trapezoidal",
            geometry_mode="thin-shell",
            compress_uniform_material_coefficients=(
                compress_uniform_material_coefficients
            ),
        ),
        material=material,
        source=source,
        device=device,
        dtype=dtype,
        compile_step=compile_step,
        torch_threads=torch_threads,
        mesh=mesh,
    )


def record_validation_traces(
    simulation: GeodesicFDTD,
    *,
    steps: int = PAPER_TRACE_STEPS,
    synchronize_every: int = 128,
    diagnostics_every: int = 512,
    recorder: Any | None = None,
) -> ValidationTraces:
    """Record barycentrically interpolated ``Er`` at A, A′, B, and B′."""

    if simulation.steps != 0:
        raise ValueError("validation recording requires a fresh simulation")
    if steps < 1:
        raise ValueError("steps must be positive")
    distributions = [
        geographic_distribution(
            simulation, 0.0, receiver.longitude_deg, receiver.altitude_m
        )
        for receiver in PAPER_RECEIVERS
    ]
    vertices = np.stack([item[0] for item in distributions])
    layers = np.asarray([item[1] for item in distributions], dtype=np.int64)
    weights = np.stack([item[2] for item in distributions])
    if recorder is None:
        values = simulation.to_numpy(
            simulation.record_er_observations(
                vertices,
                layers,
                weights,
                steps,
                synchronize_every=synchronize_every,
            )
        ).astype(np.float64, copy=False)
    else:
        values = record_er_observations_with_diagnostics(
            simulation,
            vertices,
            layers,
            weights,
            tuple(receiver.label for receiver in PAPER_RECEIVERS),
            steps,
            diagnostics_every=diagnostics_every,
            recorder=recorder,
            synchronize_every=synchronize_every,
        )
    time_steps = np.arange(steps + 1, dtype=np.int64)
    return ValidationTraces(
        time_steps=time_steps,
        time_s=time_steps.astype(np.float64) * simulation.time_step_s,
        er_v_m=values,
        labels=tuple(receiver.label for receiver in PAPER_RECEIVERS),
    )


def _receiver_spectra(
    traces: ValidationTraces,
    *,
    n_fft: int = PAPER_DFT_SIZE,
    truncations: Mapping[str, int] | None = None,
    spectral_window: str = "rectangular",
    taper_fraction: float = 0.1,
) -> tuple[dict[str, int], dict[str, NDArray[np.complex128]]]:
    if spectral_window not in {"rectangular", "cosine-tail"}:
        raise ValueError(
            "spectral_window must be 'rectangular' or 'cosine-tail'"
        )
    if not 0.0 < taper_fraction <= 1.0:
        raise ValueError("taper_fraction must be in (0, 1]")
    selected_truncations = dict(
        find_dft_truncations(traces) if truncations is None else truncations
    )
    missing = set(traces.labels) - set(selected_truncations)
    if missing:
        raise ValueError(f"missing DFT truncations for: {', '.join(sorted(missing))}")
    required = max(selected_truncations.values())
    if n_fft < required:
        raise ValueError(f"n_fft must be at least {required}")
    spectra: dict[str, NDArray[np.complex128]] = {}
    for label in traces.labels:
        cutoff = selected_truncations[label]
        if not 2 <= cutoff <= len(traces.time_steps):
            raise ValueError(
                f"DFT truncation for {label} must be between 2 and "
                f"{len(traces.time_steps)}"
            )
        signal = traces.trace(label)[:cutoff].copy()
        if spectral_window == "cosine-tail":
            taper_count = min(
                cutoff,
                max(2, int(np.ceil(taper_fraction * cutoff))),
            )
            phase = np.linspace(0.0, np.pi, taper_count)
            signal[-taper_count:] *= 0.5 * (1.0 + np.cos(phase))
        spectra[label] = np.fft.rfft(signal, n=n_fft)
    return selected_truncations, spectra


def compute_attenuation(
    traces: ValidationTraces,
    *,
    time_step_s: float = PAPER_TIME_STEP_S,
    n_fft: int = PAPER_DFT_SIZE,
    truncations: Mapping[str, int] | None = None,
    spectral_window: str = "rectangular",
    taper_fraction: float = 0.1,
) -> AttenuationCurves:
    """Compute attenuation using receiver-specific truncated DFT windows.

    By default, each record is truncated at its own positive-to-negative zero
    crossing following the primary pulse and overshoot.  This implements the
    physical windowing criterion described by Simpson and Taflove instead of
    assuming that their reported sample numbers apply to a different waveform.
    A terminal cosine taper is available only as a leakage diagnostic.
    """

    selected_truncations, complex_spectra = _receiver_spectra(
        traces,
        n_fft=n_fft,
        truncations=truncations,
        spectral_window=spectral_window,
        taper_fraction=taper_fraction,
    )
    spectra = {label: np.abs(values) for label, values in complex_spectra.items()}
    tiny = np.finfo(np.float64).tiny
    path_distance_mm = (0.25 * np.pi * EARTH_RADIUS_M) / 1.0e6
    path_ab = 20.0 * np.log10(
        np.maximum(spectra["A"], tiny) / np.maximum(spectra["B"], tiny)
    ) / path_distance_mm
    path_apbp = 20.0 * np.log10(
        np.maximum(spectra["A′"], tiny) / np.maximum(spectra["B′"], tiny)
    ) / path_distance_mm
    frequency = np.fft.rfftfreq(n_fft, d=time_step_s)
    benchmark = bannister_figure_8_guide(frequency)
    return AttenuationCurves(
        frequency,
        path_ab,
        path_apbp,
        benchmark,
        selected_truncations,
    )


def compute_phase_velocity(
    traces: ValidationTraces,
    *,
    time_step_s: float = PAPER_TIME_STEP_S,
    n_fft: int = PAPER_DFT_SIZE,
    truncations: Mapping[str, int] | None = None,
    spectral_window: str = "rectangular",
    taper_fraction: float = 0.1,
) -> PhaseVelocityCurves:
    """Compute phase velocity over the extra 45-degree receiver path."""

    selected_truncations, spectra = _receiver_spectra(
        traces,
        n_fft=n_fft,
        truncations=truncations,
        spectral_window=spectral_window,
        taper_fraction=taper_fraction,
    )
    dft_frequency = np.fft.rfftfreq(n_fft, d=time_step_s)
    frequency = paper_evaluation_frequencies()
    stop = int(np.searchsorted(dft_frequency, frequency[-1], side="right")) + 1
    path_distance_m = 0.25 * np.pi * EARTH_RADIUS_M

    def path_velocity(near: str, far: str) -> FloatArray:
        phase = np.unwrap(
            np.angle(spectra[near][:stop] * np.conj(spectra[far][:stop]))
        )
        sampled_phase = np.interp(frequency, dft_frequency[:stop], phase)
        if np.any(sampled_phase <= 0.0):
            raise ValueError("receiver phase delay must be positive")
        return (
            2.0 * np.pi * frequency * path_distance_m / sampled_phase / C_0
        )

    return PhaseVelocityCurves(
        frequency_hz=frequency,
        path_ab_fraction_c=path_velocity("A", "B"),
        path_apbp_fraction_c=path_velocity("A′", "B′"),
        benchmark_fraction_c=bannister_phase_velocity_fraction_c(frequency),
        dft_truncations=selected_truncations,
    )


def find_dft_truncations(traces: ValidationTraces) -> dict[str, int]:
    """Locate the post-overshoot zero crossing that precedes each slow tail."""

    truncations: dict[str, int] = {}
    for label in traces.labels:
        signal = traces.trace(label)
        negative_peak = int(np.argmin(signal))
        negative_amplitude = abs(float(signal[negative_peak]))
        if negative_amplitude == 0.0:
            raise ValueError(f"{label} trace has no negative primary pulse")
        positive_threshold = max(
            1.0e-6 * negative_amplitude,
            np.finfo(np.float64).eps * negative_amplitude,
        )
        overshoot_candidates = np.flatnonzero(
            signal[negative_peak + 1 :] > positive_threshold
        )
        if not len(overshoot_candidates):
            raise ValueError(
                f"{label} trace has no positive overshoot after its primary pulse; "
                "the paper's DFT window is undefined"
            )
        overshoot = negative_peak + 1 + int(overshoot_candidates[0])
        crossings = np.flatnonzero(
            (signal[overshoot:-1] >= 0.0) & (signal[overshoot + 1 :] < 0.0)
        )
        if not len(crossings):
            raise ValueError(
                f"{label} trace has no post-overshoot zero crossing before the "
                "slow tail"
            )
        crossing = overshoot + int(crossings[0])
        truncations[label] = crossing + 1
    return truncations


def _bannister_reflection_heights(
    frequency_hz: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    safe_frequency = np.where(frequency > 0.0, frequency, 1.0)
    h0_km = BANNISTER_REFERENCE_HEIGHT_KM - BANNISTER_SCALE_HEIGHT_KM * np.log(
        2.5e5 / (2.0 * np.pi * safe_frequency)
    )
    h1_km = h0_km + 2.0 * BANNISTER_SCALE_HEIGHT_KM * np.log(
        2.39e4 / (safe_frequency * BANNISTER_SCALE_HEIGHT_KM)
    )
    return h0_km, h1_km


def bannister_figure_8_guide(frequency_hz: FloatArray) -> FloatArray:
    """Evaluate the daytime attenuation model used for Figure 8.

    This implements Bannister (1984), equations (5), (7), and (8), with
    ``H = 70 km`` and ``xi_0 = xi_1 = 1 / 0.3 km`` as specified below those
    equations. Simpson and Taflove cite that curve as the previous results in
    their Figure 8.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    safe_frequency = np.where(frequency > 0.0, frequency, 1.0)
    h0_km, h1_km = _bannister_reflection_heights(frequency)
    attenuation = (
        0.143
        * safe_frequency
        * np.sqrt(h1_km / h0_km)
        * BANNISTER_SCALE_HEIGHT_KM
        * (1.0 / h0_km + 1.0 / h1_km)
    )
    return np.where(frequency > 0.0, attenuation, 0.0)


def bannister_phase_velocity_fraction_c(frequency_hz: FloatArray) -> FloatArray:
    """Evaluate Bannister (1984), equation (4), as a fraction of light speed."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    h0_km, h1_km = _bannister_reflection_heights(frequency)
    velocity = 1.0 / (0.985 * np.sqrt(h1_km / h0_km))
    return np.where(frequency > 0.0, velocity, 0.0)


def paper_evaluation_frequencies() -> FloatArray:
    """Return the 45 DFT frequencies implied by the Figure 8 markers."""

    return np.asarray(PAPER_EVALUATION_FREQUENCIES_HZ, dtype=np.float64)


def sample_paper_comparison(
    curves: AttenuationCurves,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sample attenuation curves at the fixed Figure 8 comparison bins."""

    frequency = paper_evaluation_frequencies()
    if (
        frequency[0] < curves.frequency_hz[0]
        or frequency[-1] > curves.frequency_hz[-1]
    ):
        raise ValueError("attenuation curves do not cover the paper frequencies")
    return (
        frequency,
        np.interp(frequency, curves.frequency_hz, curves.path_ab_db_per_mm),
        np.interp(frequency, curves.frequency_hz, curves.path_apbp_db_per_mm),
        bannister_figure_8_guide(frequency),
    )


def render_receiver_grid(
    output: str | Path,
    *,
    display_subdivision: int = 4,
    production_subdivision: int = 8,
    study_label: str = "Simpson–Taflove (2004)",
) -> Path:
    """Render the exact paper receiver coordinates on the polar dual grid."""

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    if display_subdivision < 0:
        raise ValueError("display subdivision must be non-negative")
    if production_subdivision < 0:
        raise ValueError("production subdivision must be non-negative")
    if not study_label:
        raise ValueError("study label must not be empty")
    mesh = build_geodesic_mesh(
        subdivision=display_subdivision,
        orientation="polar",
    )
    centers = mesh.face_centers
    longitude = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0]))
    latitude = np.rad2deg(
        np.arctan2(centers[:, 2], np.hypot(centers[:, 0], centers[:, 1]))
    )
    dual_segments = np.stack(
        (
            np.column_stack(
                (
                    longitude[mesh.edge_left_faces],
                    latitude[mesh.edge_left_faces],
                )
            ),
            np.column_stack(
                (
                    longitude[mesh.edge_right_faces],
                    latitude[mesh.edge_right_faces],
                )
            ),
        ),
        axis=1,
    )
    # Cartopy wraps the projection itself. Omitting only the longitude-seam
    # segments avoids drawing a few dual edges across the entire map.
    dual_segments = dual_segments[
        np.abs(dual_segments[:, 0, 0] - dual_segments[:, 1, 0]) < 180.0
    ]

    figure = plt.figure(figsize=(14.0, 8.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(2.2, 1.0))
    axes = (
        figure.add_subplot(
            grid[0], projection=ccrs.Mollweide(central_longitude=-47.0)
        ),
        figure.add_subplot(grid[1], projection=ccrs.PlateCarree()),
    )
    label_offsets = {
        "B′": (-6.0, 3.2),
        "A′": (-5.0, -5.0),
        "Source": (-8.0, 3.2),
        "A": (-1.0, -5.0),
        "B": (-1.0, 3.2),
    }
    locations = [("Source", PAPER_SOURCE_LONGITUDE_DEG)] + [
        (receiver.label, receiver.longitude_deg) for receiver in PAPER_RECEIVERS
    ]
    path_endpoints = {
        receiver.direction: receiver.longitude_deg
        for receiver in PAPER_RECEIVERS
        if receiver.fraction_to_antipode == 0.50
    }
    for index, axis in enumerate(axes):
        axis.set_facecolor("#f8fafc")
        axis.add_collection(
            LineCollection(
                dual_segments,
                colors="#64748b",
                linewidths=0.30,
                alpha=0.42,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
        )
        axis.coastlines(
            resolution="110m", color="#334155", linewidth=0.65, zorder=2
        )
        if index == 0:
            axis.set_global()
            axis.gridlines(color="#94a3b8", linewidth=0.4, alpha=0.5)
        else:
            axis.set_extent((-150.0, 55.0, -13.0, 13.0), crs=ccrs.PlateCarree())
            gridlines = axis.gridlines(
                draw_labels=True,
                xlocs=np.arange(-150.0, 61.0, 15.0),
                ylocs=(-10.0, 0.0, 10.0),
                color="#94a3b8",
                linewidth=0.45,
                alpha=0.55,
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.xlabel_style = {"size": 9}
            gridlines.ylabel_style = {"size": 9}

        axis.plot(
            (PAPER_SOURCE_LONGITUDE_DEG, path_endpoints["east"]),
            (0.0, 0.0),
            color="#2563eb",
            linewidth=3.0,
            alpha=0.75,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        axis.plot(
            (PAPER_SOURCE_LONGITUDE_DEG, path_endpoints["west"]),
            (0.0, 0.0),
            color="#d97706",
            linewidth=3.0,
            alpha=0.75,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        axis.scatter(
            (PAPER_SOURCE_LONGITUDE_DEG,),
            (PAPER_SOURCE_LATITUDE_DEG,),
            s=150 if index == 0 else 180,
            marker="*",
            color="#dc2626",
            edgecolors="white",
            linewidths=1.1,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        for receiver in PAPER_RECEIVERS:
            color = "#2563eb" if receiver.direction == "east" else "#d97706"
            marker = "o" if receiver.fraction_to_antipode == 0.25 else "s"
            axis.scatter(
                (receiver.longitude_deg,),
                (0.0,),
                s=75 if index == 0 else 95,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=1.2,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        for label, receiver_longitude in locations:
            longitude_offset, latitude_offset = label_offsets[label]
            axis.text(
                receiver_longitude + longitude_offset,
                latitude_offset,
                f"{label}\n({receiver_longitude:g}°)",
                fontsize=10 if index == 0 else 11,
                fontweight="bold",
                ha="left",
                va="center",
                color="#0f172a",
                transform=ccrs.PlateCarree(),
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )

    axes[0].set_title("Global receiver geometry", fontsize=16, fontweight="bold")
    axes[1].set_title(
        "Equatorial detail: exact requested coordinates on the dual grid",
        fontsize=14,
        fontweight="bold",
    )
    legend = (
        Line2D(
            (0,),
            (0,),
            marker="*",
            color="none",
            markerfacecolor="#dc2626",
            markeredgecolor="white",
            markersize=14,
            label="Source (47° W)",
        ),
        Line2D(
            (0,),
            (0,),
            marker="o",
            color="#2563eb",
            markersize=8,
            label="A: east 45°",
        ),
        Line2D(
            (0,),
            (0,),
            marker="s",
            color="#2563eb",
            markersize=8,
            label="B: east 90°",
        ),
        Line2D(
            (0,),
            (0,),
            marker="o",
            color="#d97706",
            markersize=8,
            label="A′: west 45°",
        ),
        Line2D(
            (0,),
            (0,),
            marker="s",
            color="#d97706",
            markersize=8,
            label="B′: west 90°",
        ),
        Line2D((0,), (0,), color="#64748b", label="Dual-cell boundary"),
    )
    axes[0].legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=True,
        fontsize=10,
    )
    figure.suptitle(
        f"{study_label} receiver locations on the polar geodesic grid",
        fontsize=19,
        fontweight="bold",
    )
    production_cell_count = 10 * 4**production_subdivision + 2
    figure.text(
        0.5,
        0.006,
        f"Display grid: subdivision {display_subdivision} "
        f"({mesh.n_vertices:,} dual cells). Production subdivision "
        f"{production_subdivision} has the same polar orientation and recursive "
        f"topology but {production_cell_count:,} cells.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def render_figure_7(
    traces: ValidationTraces, output: str | Path
) -> Path:
    """Render the two temporal-response panels of Figure 7."""

    import matplotlib.pyplot as plt

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.4),
        sharex=True,
        constrained_layout=True,
    )
    for ax, near, far, fraction in (
        (axes[0], "A", "A′", "1/4 distance to antipode"),
        (axes[1], "B", "B′", "1/2 distance to antipode"),
    ):
        ax.plot(
            traces.time_steps,
            1.0e6 * traces.trace(near),
            label=f"{near} (east)",
        )
        ax.plot(
            traces.time_steps,
            1.0e6 * traces.trace(far),
            label=f"{far} (west)",
        )
        ax.axhline(0.0, color="0.35", linewidth=0.7)
        ax.set_ylabel(r"$E_r$ ($\mu$V/m)")
        ax.set_title(f"Equator, {fraction}")
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("Time steps (Δt = 3 μs)")
    figure.suptitle("Simpson & Taflove (2004), Fig. 7 verification run")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def render_figure_8(
    curves: AttenuationCurves, output: str | Path
) -> Path:
    """Render attenuation curves and the Bannister daytime model."""

    import matplotlib.pyplot as plt

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_frequency, path_ab, path_apbp, _ = sample_paper_comparison(curves)
    guide_frequency = np.geomspace(5.0, 2_000.0, 512)
    x_ticks = (5, 10, 20, 50, 100, 200, 500, 1_000, 2_000)
    y_ticks = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30)
    figure, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    ax.axvspan(
        *curves.valid_frequency_hz,
        color="#e8f2ff",
        label="paper-valid DFT window",
    )
    ax.plot(
        guide_frequency,
        bannister_figure_8_guide(guide_frequency),
        color="black",
        label="Bannister daytime model",
    )
    ax.plot(
        comparison_frequency,
        path_ab,
        "*",
        markersize=6,
        label="Path A–B",
    )
    ax.plot(
        comparison_frequency,
        path_apbp,
        ".",
        markersize=5,
        label="Path A′–B′",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5.0, 2_000.0)
    ax.set_ylim(0.1, 30.0)
    ax.set_xticks(x_ticks, labels=[f"{value:g}" for value in x_ticks])
    ax.set_yticks(y_ticks, labels=[f"{value:g}" for value in y_ticks])
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Attenuation rate (dB/Mm)", fontsize=20)
    ax.set_title(
        "Simpson & Taflove (2004), Fig. 8 verification run",
        fontsize=20,
    )
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, which="both", color="0.9", linewidth=0.6)
    ax.legend(fontsize=18)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def validation_metrics(curves: AttenuationCurves) -> dict[str, float]:
    """Return errors from the Bannister model at the paper's DFT bins."""

    frequency, path_ab, path_apbp, benchmark = sample_paper_comparison(curves)
    ab_residual = path_ab - benchmark
    apbp_residual = path_apbp - benchmark
    ab_maximum_index = int(np.argmax(np.abs(ab_residual)))
    apbp_maximum_index = int(np.argmax(np.abs(apbp_residual)))
    return {
        "path_ab_mean_absolute_error_db_per_mm": float(
            np.mean(np.abs(ab_residual))
        ),
        "path_apbp_mean_absolute_error_db_per_mm": float(
            np.mean(np.abs(apbp_residual))
        ),
        "path_ab_maximum_absolute_error_db_per_mm": float(
            np.max(np.abs(ab_residual))
        ),
        "path_apbp_maximum_absolute_error_db_per_mm": float(
            np.max(np.abs(apbp_residual))
        ),
        "path_ab_maximum_error_frequency_hz": float(
            frequency[ab_maximum_index]
        ),
        "path_apbp_maximum_error_frequency_hz": float(
            frequency[apbp_maximum_index]
        ),
        "path_ab_maximum_residual_db_per_mm": float(
            ab_residual[ab_maximum_index]
        ),
        "path_apbp_maximum_residual_db_per_mm": float(
            apbp_residual[apbp_maximum_index]
        ),
    }


def phase_velocity_metrics(curves: PhaseVelocityCurves) -> dict[str, float]:
    """Return phase-velocity errors relative to Bannister equation (4)."""

    ab_residual = curves.path_ab_fraction_c - curves.benchmark_fraction_c
    apbp_residual = curves.path_apbp_fraction_c - curves.benchmark_fraction_c
    return {
        "path_ab_phase_velocity_mean_absolute_error_fraction_c": float(
            np.mean(np.abs(ab_residual))
        ),
        "path_apbp_phase_velocity_mean_absolute_error_fraction_c": float(
            np.mean(np.abs(apbp_residual))
        ),
        "path_ab_phase_velocity_maximum_absolute_error_fraction_c": float(
            np.max(np.abs(ab_residual))
        ),
        "path_apbp_phase_velocity_maximum_absolute_error_fraction_c": float(
            np.max(np.abs(apbp_residual))
        ),
    }


def arrival_metrics(traces: ValidationTraces) -> dict[str, float | int]:
    """Return negative-peak travel times and apparent pulse velocities."""

    result: dict[str, float | int] = {}
    peaks = {label: int(np.argmin(traces.trace(label))) for label in traces.labels}
    source_center_s = PAPER_SOURCE_CENTER_STEPS * PAPER_TIME_STEP_S
    receivers = {receiver.label: receiver for receiver in PAPER_RECEIVERS}
    for label, peak_index in peaks.items():
        travel_time_s = float(traces.time_s[peak_index] - source_center_s)
        if travel_time_s <= 0.0:
            raise ValueError(f"{label} peak must follow the source center")
        distance_m = receivers[label].fraction_to_antipode * np.pi * EARTH_RADIUS_M
        result[f"{label}_negative_peak_travel_time_s"] = travel_time_s
        result[f"{label}_apparent_peak_velocity_fraction_c"] = float(
            distance_m / travel_time_s / C_0
        )
    for near, far, name in (("A", "B", "path_ab"), ("A′", "B′", "path_apbp")):
        travel_time_s = float(traces.time_s[peaks[far]] - traces.time_s[peaks[near]])
        if travel_time_s <= 0.0:
            raise ValueError(f"{far} peak must follow the {near} peak")
        distance_m = 0.25 * np.pi * EARTH_RADIUS_M
        result[f"{name}_negative_peak_travel_time_s"] = travel_time_s
        result[f"{name}_apparent_peak_velocity_fraction_c"] = float(
            distance_m / travel_time_s / C_0
        )
    return result


def source_distribution_metrics(
    simulation: GeodesicFDTD,
) -> dict[str, float | int]:
    """Describe the represented radial position of the validation source."""

    if simulation.source is None:
        return {}
    _, layers, weights = simulation.source.staggered_distribution(simulation)
    active_altitudes = simulation.altitudes_m[layers]
    return {
        "source_requested_altitude_m": float(simulation.source.altitude_m),
        "source_staggered_centroid_altitude_m": float(
            np.sum(weights * active_altitudes)
        ),
        "source_staggered_lower_plane_altitude_m": float(
            np.min(active_altitudes)
        ),
        "source_staggered_upper_plane_altitude_m": float(
            np.max(active_altitudes)
        ),
        "source_staggered_radial_support_planes": int(len(np.unique(layers))),
        "source_distribution_weight_sum": float(np.sum(weights)),
    }


def trace_metrics(traces: ValidationTraces) -> dict[str, float | int]:
    """Summarize pulse arrival and east/west asymmetry in Figure 7 traces."""

    result: dict[str, float | int] = {}
    for label in traces.labels:
        values = traces.trace(label)
        peak_index = int(np.argmin(values))
        result[f"{label}_negative_peak_step"] = int(traces.time_steps[peak_index])
        result[f"{label}_negative_peak_uv_m"] = float(1.0e6 * values[peak_index])
    for east, west, name in (("A", "A′", "quarter"), ("B", "B′", "half")):
        east_values = traces.trace(east)
        west_values = traces.trace(west)
        scale = max(float(np.sqrt(np.mean(east_values**2))), np.finfo(float).tiny)
        result[f"{name}_east_west_relative_rms"] = float(
            np.sqrt(np.mean((east_values - west_values) ** 2)) / scale
        )
    return result

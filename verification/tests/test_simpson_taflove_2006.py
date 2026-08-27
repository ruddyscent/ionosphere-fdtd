import importlib.util
import json

import numpy as np
import pytest

from verification.simpson_taflove_2004.materials import ETOPO5Relief
from verification.simpson_taflove_2004.model import ValidationTraces
from verification.simpson_taflove_2006.materials import HermanceFigure15Material
from verification.simpson_taflove_2006.model import (
    PAPER_ENVELOPE_FWHM_S,
    PAPER_OIL_AREA_KM2,
    PAPER_OIL_CONDUCTIVITY_FACTOR,
    PAPER_OIL_MEDIAN_DEPTH_M,
    PAPER_OIL_RADIUS_M,
    PAPER_OIL_THICKNESS_M,
    THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
    THESIS_DAYTIME_EFFECTIVE_REFLECTION_HEIGHT_M,
    THESIS_NIGHTTIME_EFFECTIVE_REFLECTION_HEIGHT_M,
    THESIS_NIGHTTIME_IONOSPHERE_REFERENCE_HEIGHT_M,
    THESIS_NIGHTTIME_IONOSPHERE_SCALE_HEIGHT_M,
    THESIS_OIL_MAXIMUM_BACKGROUND_CONDUCTIVITY_S_M,
    DayNightHemisphereProfile,
    RadarResolutionConvergence,
    RadarTraces,
    _surface_h_distributions,
    build_paper_adaptive_mesh,
    compare_radar_resolution_pairs,
    compute_radar_perturbation,
    create_radar_simulation,
    load_radar_traces,
    normalized_figure_5_traces,
    paper_anomalies,
    radar_field_metrics,
    radar_radial_altitudes_m,
    record_radar_traces,
    save_radar_traces,
)
from verification.simpson_taflove_2006.__main__ import main

requires_natural_earth = pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("cartopy", "shapely")
    ),
    reason="Natural Earth materials require: uv sync --extra visualization",
)


def test_figure_5_normalization_preserves_four_individual_records() -> None:
    time = np.arange(3, dtype=np.float64)
    values = np.asarray(
        (
            (0.0, 0.0, 0.0, 0.0),
            (-2.0, -1.0, -0.5, -0.25),
            (1.0, 0.5, 0.25, 0.125),
        )
    )
    traces = ValidationTraces(
        time_steps=np.arange(3, dtype=np.int64),
        time_s=time,
        er_v_m=values,
        labels=("A", "A′", "B", "B′"),
    )

    normalized = normalized_figure_5_traces(traces)

    assert tuple(normalized) == ("A", "A′", "B", "B′")
    np.testing.assert_allclose(normalized["A"], -values[:, 0] / 2.0)
    np.testing.assert_allclose(normalized["A′"], -values[:, 1] / 2.0)
    np.testing.assert_allclose(normalized["B"], -values[:, 2] / 2.0)
    np.testing.assert_allclose(normalized["B′"], -values[:, 3] / 2.0)


def test_paper_oil_geometry_matches_area_depth_and_contrast() -> None:
    oil = paper_anomalies(include_oil=True)[1]

    assert np.pi * (PAPER_OIL_RADIUS_M / 1_000.0) ** 2 == pytest.approx(
        PAPER_OIL_AREA_KM2
    )
    assert oil.altitude_max_m - oil.altitude_min_m == pytest.approx(
        PAPER_OIL_THICKNESS_M
    )
    assert -0.5 * (oil.altitude_max_m + oil.altitude_min_m) == pytest.approx(
        PAPER_OIL_MEDIAN_DEPTH_M
    )
    assert oil.conductivity_factor == PAPER_OIL_CONDUCTIVITY_FACTOR
    assert oil.maximum_background_conductivity_s_m == (
        THESIS_OIL_MAXIMUM_BACKGROUND_CONDUCTIVITY_S_M
    )
    assert oil.target_area_m2 == PAPER_OIL_AREA_KM2 * 1.0e6


def test_paper_adaptive_mesh_refines_transmitter_and_oil_on_one_surface() -> None:
    mesh = build_paper_adaptive_mesh(
        2,
        base_subdivision=1,
        core_radius_deg=8.0,
        transition_width_deg=8.0,
    )
    targets = (
        (46.5, -90.9, "transmitter"),
        (69.0, -156.0, "oil-receiver"),
    )

    for latitude, longitude, _ in targets:
        latitude_rad = np.deg2rad(latitude)
        longitude_rad = np.deg2rad(longitude)
        direction = np.asarray(
            (
                np.cos(latitude_rad) * np.cos(longitude_rad),
                np.cos(latitude_rad) * np.sin(longitude_rad),
                np.sin(latitude_rad),
            )
        )
        nearest = int(np.argmax(mesh.face_centers @ direction))
        assert mesh.face_levels[nearest] == 2
    assert [
        region["label"] for region in mesh.refinement_spec["regions"]
    ] == [target[2] for target in targets]
    assert mesh.n_faces < 20 * 4**2


@requires_natural_earth
def test_adaptive_reference_and_anomaly_runs_share_exact_mesh_signature() -> None:
    mesh = build_paper_adaptive_mesh(
        1,
        base_subdivision=0,
        core_radius_deg=12.0,
        transition_width_deg=12.0,
    )
    reference_simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        mesh=mesh,
    )
    anomaly_simulation = create_radar_simulation(
        include_oil=True,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        mesh=mesh,
    )

    reference = record_radar_traces(
        reference_simulation, steps=1, case="reference"
    )
    anomaly = record_radar_traces(anomaly_simulation, steps=1, case="anomaly")

    assert reference.run_signature == anomaly.run_signature
    np.testing.assert_array_equal(
        reference_simulation.mesh.faces, anomaly_simulation.mesh.faces
    )


def test_radar_resolution_comparison_interpolates_paired_fields() -> None:
    def traces(time: np.ndarray, case: str, signature: str) -> RadarTraces:
        hr = np.sin(2.0 * np.pi * 5.0 * time)
        east = np.cos(2.0 * np.pi * 5.0 * time)
        north = 0.5 * hr
        if case == "anomaly":
            hr = 1.1 * hr
            east = 1.2 * east
            north = 1.3 * north
        return RadarTraces(time, hr, east, north, 0.0, case, signature)

    coarse_time = np.linspace(-0.01, 0.09, 101)
    fine_time = np.linspace(-0.01, 0.09, 201)
    result = compare_radar_resolution_pairs(
        traces(coarse_time, "reference", "coarse"),
        traces(coarse_time, "anomaly", "coarse"),
        traces(fine_time, "reference", "fine"),
        traces(fine_time, "anomaly", "fine"),
        coarse_target_subdivision=9,
        fine_target_subdivision=10,
        relative_stop_s=0.085,
    )

    assert isinstance(result, RadarResolutionConvergence)
    assert result.samples == 171
    for name in (
        "reference_hr_relative_l2",
        "reference_ht_relative_l2",
        "anomaly_hr_relative_l2",
        "anomaly_ht_relative_l2",
        "perturbation_hr_relative_l2",
        "perturbation_ht_relative_l2",
    ):
        assert getattr(result, name) < 3.0e-4


def test_radar_resolution_comparison_clips_to_common_half_step_window() -> None:
    def traces(time: np.ndarray, case: str, signature: str) -> RadarTraces:
        values = np.sin(2.0 * np.pi * 5.0 * time)
        if case == "anomaly":
            values = 1.1 * values
        return RadarTraces(time, values, values, values, 0.0, case, signature)

    coarse_time = np.linspace(-0.01, 0.0849998, 101)
    fine_time = np.linspace(-0.01, 0.0849999, 201)
    result = compare_radar_resolution_pairs(
        traces(coarse_time, "reference", "coarse"),
        traces(coarse_time, "anomaly", "coarse"),
        traces(fine_time, "reference", "fine"),
        traces(fine_time, "anomaly", "fine"),
        coarse_target_subdivision=9,
        fine_target_subdivision=10,
    )

    assert result.comparison_stop_s <= coarse_time[-1]


def test_radar_resolution_comparison_rejects_unpaired_mesh_runs() -> None:
    time = np.linspace(0.0, 0.1, 11)
    values = np.sin(2.0 * np.pi * 5.0 * time)
    reference = RadarTraces(
        time, values, values, values, 0.0, "reference", "mesh-a"
    )
    anomaly = RadarTraces(
        time, values, values, values, 0.0, "anomaly", "mesh-b"
    )

    with pytest.raises(ValueError, match="share one run signature"):
        compare_radar_resolution_pairs(
            reference,
            anomaly,
            reference,
            anomaly,
            coarse_target_subdivision=9,
            fine_target_subdivision=10,
        )


def test_adaptive_analysis_writes_screening_verdict(tmp_path) -> None:
    def traces(time: np.ndarray, case: str, level: int) -> RadarTraces:
        scale = 1.0 if level == 10 else 1.01
        hr = scale * np.sin(2.0 * np.pi * 5.0 * time)
        east = scale * np.cos(2.0 * np.pi * 5.0 * time)
        north = 0.5 * hr
        if case == "anomaly":
            hr *= 1.1
            east *= 1.2
            north *= 1.3
        signature = json.dumps(
            {
                "runtime": "torch",
                "dtype": "float32",
                "git_revision": "test",
                "mesh_vertices_sha256": f"vertices-{level}",
                "mesh_faces_sha256": f"faces-{level}",
                "time_step_s": float(time[1] - time[0]),
            }
        )
        return RadarTraces(time, hr, east, north, 0.0, case, signature)

    input_dir = tmp_path / "traces"
    for level, samples in ((9, 101), (10, 201)):
        time = np.linspace(-0.01, 0.09, samples)
        for case in ("reference", "anomaly"):
            save_radar_traces(
                traces(time, case, level), input_dir / f"s{level}-{case}.npz"
            )
    summary = tmp_path / "summary.json"

    assert (
        main(
            [
                "analyze-adaptive",
                "--input-dir",
                str(input_dir),
                "--summary",
                str(summary),
                "--relative-l2-threshold",
                "0.02",
            ]
        )
        == 0
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))

    assert payload["screening"]["runtime"] == "torch"
    assert payload["screening"]["dtype"] == "float32"
    assert payload["screening"]["relative_l2_threshold"] == 0.02
    assert payload["screening"]["maximum_relative_l2"] < 0.011
    assert payload["screening"]["converged"] is True
    assert set(payload["fine_metrics"]) == {"peak", "pointwise"}


@requires_natural_earth
def test_conservative_radar_anomaly_preserves_oil_area_on_both_grids() -> None:
    simulation = create_radar_simulation(
        include_oil=True,
        subdivision=3,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    oil_index = len(simulation.material.anomalies) - 1
    er_fraction = simulation.anomaly_horizontal_fractions_er[oil_index]
    et_fraction = simulation.anomaly_horizontal_fractions_et[oil_index]
    radius_squared = simulation.config.earth_radius_m**2
    represented_er_km2 = (
        er_fraction @ simulation.mesh.dual_cell_solid_angles * radius_squared / 1.0e6
    )
    represented_et_km2 = (
        et_fraction @ simulation.mesh.edge_diamond_solid_angles()
        * radius_squared
        / 1.0e6
    )

    assert represented_er_km2 == pytest.approx(PAPER_OIL_AREA_KM2)
    assert represented_et_km2 == pytest.approx(PAPER_OIL_AREA_KM2)
    assert simulation.material.deep_rock_resistivity_ohm_m == 500.0
    assert np.count_nonzero(er_fraction) >= 1
    assert np.count_nonzero(et_fraction) >= 1


def test_paper_anomalies_can_omit_or_resize_shield() -> None:
    without_shield = paper_anomalies(include_oil=False, include_shield=False)
    resized = paper_anomalies(
        include_oil=False, include_shield=True, shield_radius_m=1_500_000.0
    )

    assert without_shield == ()
    assert len(resized) == 1
    assert resized[0].radius_m == 1_500_000.0


def test_radar_grid_refines_lithosphere_to_1_25_km() -> None:
    altitudes = np.asarray(radar_radial_altitudes_m())

    np.testing.assert_allclose(
        altitudes[(altitudes >= -5_000.0) & (altitudes <= 0.0)],
        (-5_000.0, -3_750.0, -2_500.0, -1_250.0, 0.0),
    )
    assert len(altitudes) - 1 == 43


def test_day_night_profile_uses_declared_solar_hemispheres() -> None:
    profile = DayNightHemisphereProfile(
        daytime_value=70_000.0,
        nighttime_value=92_800.0,
        subsolar_latitude_deg=0.0,
        subsolar_longitude_deg=0.0,
    )

    values = profile(np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))))

    np.testing.assert_array_equal(values, (70_000.0, 92_800.0))


def test_dawn_zero_geometry_places_noon_at_90_degrees_east() -> None:
    profile = DayNightHemisphereProfile(
        daytime_value=1.0,
        nighttime_value=0.0,
        subsolar_latitude_deg=0.0,
        subsolar_longitude_deg=THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
    )
    longitude = np.deg2rad(np.asarray((90.0, -90.0)))
    directions = np.column_stack(
        (np.cos(longitude), np.sin(longitude), np.zeros(2))
    )

    np.testing.assert_array_equal(profile(directions), (1.0, 0.0))


def test_figure_15_material_uses_distinct_land_and_ocean_layers() -> None:
    material = HermanceFigure15Material(
        land_classifier=lambda directions: directions[:, 0] > 0.0,
        ocean_depth_m=1.0,
    )
    directions = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    altitudes = np.asarray((-2_500.0, -7_500.0, -15_000.0, -30_000.0, -50_000.0))

    sigma, _ = material.sample(directions, altitudes, 6_371_000.0)

    np.testing.assert_allclose(
        1.0 / sigma[0], (10.0, 5_000.0, 5_000.0, 5_000.0, 50.0)
    )
    np.testing.assert_allclose(
        1.0 / sigma[1], (5.0, 50.0, 500.0, 200.0, 50.0)
    )


def test_bannister_profiles_match_dissertation_reflection_heights() -> None:
    material = HermanceFigure15Material(
        land_classifier=lambda directions: np.ones(len(directions), dtype=np.bool_),
        ionosphere_reference_height_sampler=DayNightHemisphereProfile(
            70_000.0,
            THESIS_NIGHTTIME_IONOSPHERE_REFERENCE_HEIGHT_M,
            subsolar_longitude_deg=THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
        ),
        ionosphere_scale_height_sampler=DayNightHemisphereProfile(
            1_000.0 / 0.3,
            THESIS_NIGHTTIME_IONOSPHERE_SCALE_HEIGHT_M,
            subsolar_longitude_deg=THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
        ),
    )
    longitude = np.deg2rad(np.asarray((90.0, -90.0)))
    directions = np.column_stack(
        (np.cos(longitude), np.sin(longitude), np.zeros(2))
    )
    altitudes = np.asarray(
        (
            THESIS_DAYTIME_EFFECTIVE_REFLECTION_HEIGHT_M,
            THESIS_NIGHTTIME_EFFECTIVE_REFLECTION_HEIGHT_M,
        )
    )

    sigma, _ = material.sample(directions, altitudes, 6_371_000.0)

    assert sigma[0, 0] == pytest.approx(sigma[1, 1], rel=0.25)
    assert sigma[0, 0] == pytest.approx(3.0e-9, rel=0.02)


@requires_natural_earth
def test_thesis_radar_setup_installs_day_night_ionosphere() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        ionosphere_model="day-night",
        subsolar_longitude_deg=0.0,
        upper_crust_resistivity_ohm_m=5_000.0,
        deep_lithosphere_resistivity_ohm_m=50.0,
    )
    day = np.asarray(((1.0, 0.0, 0.0),))
    night = np.asarray(((-1.0, 0.0, 0.0),))

    assert simulation.material.ionosphere_reference_height_sampler(day)[0] == (
        70_000.0
    )
    assert simulation.material.ionosphere_reference_height_sampler(night)[0] == (
        THESIS_NIGHTTIME_IONOSPHERE_REFERENCE_HEIGHT_M
    )
    assert simulation.material.ionosphere_scale_height_sampler(night)[0] == (
        THESIS_NIGHTTIME_IONOSPHERE_SCALE_HEIGHT_M
    )
    assert simulation.material.upper_crust_resistivity_ohm_m == 5_000.0
    assert simulation.material.deep_rock_resistivity_ohm_m == 50.0


@requires_natural_earth
def test_radar_setup_can_install_figure_15_material() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        lithosphere_profile="figure-15",
        ionosphere_model="day-night",
    )

    assert isinstance(simulation.material, HermanceFigure15Material)
    noon = np.asarray(((0.0, 1.0, 0.0),))
    midnight = np.asarray(((0.0, -1.0, 0.0),))
    assert simulation.material.ionosphere_reference_height_sampler(noon)[0] == (
        70_000.0
    )
    assert simulation.material.ionosphere_reference_height_sampler(midnight)[0] == (
        THESIS_NIGHTTIME_IONOSPHERE_REFERENCE_HEIGHT_M
    )
    assert simulation.material.anomalies[0].conductivity_factor == pytest.approx(1.2)


@requires_natural_earth
def test_radar_setup_forwards_compiled_chunk_size() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        compile_chunk_size=32,
    )

    assert simulation.compile_chunk_size == 32


@requires_natural_earth
def test_short_radar_run_records_three_surface_components() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    traces = record_radar_traces(simulation, steps=3, case="reference")

    assert simulation.source is not None
    assert simulation.source.carrier_frequency_hz == 20.0
    assert simulation.source.peak_current_a == 300.0
    assert simulation.source.azimuths_deg == (0.0, 90.0)
    assert simulation.source.line_lengths_m == (22_500.0, 22_500.0)
    assert simulation.config.loss_integration == "trapezoidal"
    assert simulation.config.geometry_mode == "full-spherical"
    assert simulation.config.mesh_orientation == "polar"
    pentagons = simulation.mesh.vertices[simulation.mesh.vertex_degree == 5]
    assert np.max(pentagons[:, 2]) == pytest.approx(1.0)
    assert np.min(pentagons[:, 2]) == pytest.approx(-1.0)
    assert PAPER_ENVELOPE_FWHM_S == pytest.approx(42.5e-3)
    assert traces.hr_a_m.shape == (4,)
    assert traces.ht_east_a_m.shape == (4,)
    assert traces.ht_north_a_m.shape == (4,)
    assert traces.time_s[0] == pytest.approx(-0.5 * simulation.time_step_s)
    assert traces.time_s[1] == pytest.approx(0.5 * simulation.time_step_s)


@requires_natural_earth
def test_radar_setup_can_retain_native_orientation() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=1,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        mesh_orientation="native",
    )

    north = int(np.argmax(simulation.mesh.vertices[:, 2]))
    south = int(np.argmin(simulation.mesh.vertices[:, 2]))
    assert simulation.config.mesh_orientation == "native"
    assert simulation.mesh.vertex_degree[north] == 6
    assert simulation.mesh.vertex_degree[south] == 6


@requires_natural_earth
def test_radar_geometry_and_generated_mesh_optimization_are_configurable() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=1,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        geometry_mode="full-spherical",
        mesh_optimization_steps=1,
    )

    assert simulation.config.geometry_mode == "full-spherical"
    assert simulation.config.mesh_optimization_steps == 1


@requires_natural_earth
def test_radar_source_basis_and_altitude_are_configurable() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        source_altitude_m=-625.0,
        source_azimuths_deg=(90.0,),
    )

    assert simulation.source is not None
    assert simulation.source.altitude_m == -625.0
    assert simulation.source.azimuths_deg == (90.0,)
    assert simulation.source.line_lengths_m == (22_500.0,)


def test_etopo_radar_geometry_can_follow_local_terrain(monkeypatch) -> None:
    source_direction = np.asarray(
        (
            np.cos(np.deg2rad(46.5)) * np.cos(np.deg2rad(-90.9)),
            np.cos(np.deg2rad(46.5)) * np.sin(np.deg2rad(-90.9)),
            np.sin(np.deg2rad(46.5)),
        )
    )
    oil_direction = np.asarray(
        (
            np.cos(np.deg2rad(69.0)) * np.cos(np.deg2rad(-156.0)),
            np.cos(np.deg2rad(69.0)) * np.sin(np.deg2rad(-156.0)),
            np.sin(np.deg2rad(69.0)),
        )
    )

    class FakeRelief:
        def __call__(self, directions: np.ndarray) -> np.ndarray:
            result = np.zeros(len(directions))
            result[directions @ source_direction > 1.0 - 1.0e-12] = 236.8
            result[directions @ oil_direction > 1.0 - 1.0e-12] = 305.0
            return result

    monkeypatch.setattr(
        ETOPO5Relief,
        "from_file",
        classmethod(lambda cls, path: FakeRelief()),
    )
    terrain = create_radar_simulation(
        include_oil=True,
        subdivision=0,
        material_model="etopo5",
        etopo5_path="unused.dat",
        dtype="float64",
        compile_step=False,
        vertical_reference="terrain",
    )
    sea_level = create_radar_simulation(
        include_oil=True,
        subdivision=0,
        material_model="etopo5",
        etopo5_path="unused.dat",
        dtype="float64",
        compile_step=False,
        vertical_reference="sea-level",
    )

    terrain_oil = terrain.material.anomalies[-1]
    sea_level_oil = sea_level.material.anomalies[-1]
    assert terrain.source.altitude_m == pytest.approx(236.8)
    assert terrain.radar_receiver_altitude_m == pytest.approx(305.0)
    assert 305.0 - 0.5 * (
        terrain_oil.altitude_min_m + terrain_oil.altitude_max_m
    ) == pytest.approx(PAPER_OIL_MEDIAN_DEPTH_M)
    assert sea_level.source.altitude_m == 0.0
    assert sea_level.radar_receiver_altitude_m == 0.0
    assert -0.5 * (
        sea_level_oil.altitude_min_m + sea_level_oil.altitude_max_m
    ) == pytest.approx(PAPER_OIL_MEDIAN_DEPTH_M)


@requires_natural_earth
def test_local_linear_radar_receiver_reconstructs_target_direction() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=1,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    faces, layers, weights, *_ = _surface_h_distributions(
        simulation, receiver_support="local-linear"
    )
    unique_faces = np.unique(faces)
    horizontal_weights = np.asarray(
        [weights[faces == face].sum() for face in unique_faces]
    )
    represented = (
        horizontal_weights @ simulation.mesh.face_centers[unique_faces]
    )
    represented /= np.linalg.norm(represented)
    target = np.asarray(
        (
            np.cos(np.deg2rad(69.0)) * np.cos(np.deg2rad(-156.0)),
            np.cos(np.deg2rad(69.0)) * np.sin(np.deg2rad(-156.0)),
            np.sin(np.deg2rad(69.0)),
        )
    )
    radial_altitude = float(
        weights.ravel() @ simulation.radial_midpoint_altitudes_m[layers.ravel()]
    )

    assert len(unique_faces) == 4
    assert weights.sum() == pytest.approx(1.0)
    assert radial_altitude == pytest.approx(0.0)
    assert represented @ target == pytest.approx(1.0, abs=2.0e-3)


@requires_natural_earth
def test_default_radar_receiver_uses_local_linear_support() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=1,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )

    faces, *_ = _surface_h_distributions(simulation)

    assert len(np.unique(faces)) == 4


@requires_natural_earth
def test_radar_receiver_interpolates_both_h_components_to_terrain() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=1,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    simulation.radar_receiver_altitude_m = 305.0

    faces, hr_layers, hr_weights, edges, ht_layers, ht_weights = (
        _surface_h_distributions(simulation)
    )

    del faces, edges
    represented_hr_altitude = float(
        hr_weights.ravel()
        @ simulation.radial_midpoint_altitudes_m[hr_layers.ravel()]
    )
    assert represented_hr_altitude == pytest.approx(305.0)
    radial_count = 2
    for component in range(2):
        for offset in range(0, ht_layers.shape[1], radial_count):
            selected = slice(offset, offset + radial_count)
            weights = ht_weights[component, selected]
            represented = float(
                weights @ simulation.altitudes_m[ht_layers[component, selected]]
                / weights.sum()
            )
            assert represented == pytest.approx(305.0)


@requires_natural_earth
def test_radar_receiver_rejects_unknown_support() -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )

    with pytest.raises(ValueError, match="receiver_support"):
        _surface_h_distributions(simulation, receiver_support="unknown")


@requires_natural_earth
def test_radar_courant_factor_controls_automatic_time_step() -> None:
    conservative = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        courant_factor=0.4,
    )
    limit = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
        courant_factor=1.0,
    )

    assert limit.time_step_s == pytest.approx(2.5 * conservative.time_step_s)


def test_pointwise_radar_normalization_has_expected_db_levels() -> None:
    time = np.linspace(0.0, 0.1, 101)
    base = np.sin(2.0 * np.pi * 20.0 * time)
    reference = RadarTraces(
        time,
        base,
        base,
        np.zeros_like(base),
        0.0,
        "reference",
        "test-signature",
    )
    anomaly = RadarTraces(
        time,
        11.0 * base,
        (1.0 + 10.0 ** (-30.0 / 20.0)) * base,
        np.zeros_like(base),
        0.0,
        "anomaly",
        "test-signature",
    )

    curves = compute_radar_perturbation(
        reference,
        anomaly,
        relative_stop_s=0.1,
    )

    np.testing.assert_allclose(curves.delta_hr_db[curves.valid_hr], 20.0)
    np.testing.assert_allclose(curves.delta_ht_db[curves.valid_ht], -30.0)
    metrics = radar_field_metrics(reference, anomaly, curves)
    assert metrics["delta_hr_peak_normalized_db"] == pytest.approx(20.0)
    assert metrics["delta_ht_peak_normalized_db"] == pytest.approx(-30.0)

    peak_curves = compute_radar_perturbation(
        reference,
        anomaly,
        relative_stop_s=0.1,
        normalization="peak",
    )
    assert peak_curves.normalization == "peak"
    assert np.max(peak_curves.delta_hr_db) == pytest.approx(20.0)
    assert np.max(peak_curves.delta_ht_db) == pytest.approx(-30.0)
    assert np.all(peak_curves.valid_hr)
    assert np.all(peak_curves.valid_ht)


def test_tangential_field_definitions_are_explicit_and_distinct() -> None:
    time = np.linspace(0.0, 0.1, 11)
    reference = RadarTraces(
        time,
        np.ones_like(time),
        2.0 * np.ones_like(time),
        4.0 * np.ones_like(time),
        0.0,
        "reference",
        "test-signature",
    )
    anomaly = RadarTraces(
        time,
        np.ones_like(time),
        4.0 * np.ones_like(time),
        5.0 * np.ones_like(time),
        0.0,
        "anomaly",
        "test-signature",
    )

    east = compute_radar_perturbation(
        reference, anomaly, relative_stop_s=0.1, ht_definition="east"
    )
    north = compute_radar_perturbation(
        reference, anomaly, relative_stop_s=0.1, ht_definition="north"
    )
    vector = compute_radar_perturbation(
        reference,
        anomaly,
        relative_stop_s=0.1,
        ht_definition="vector-difference",
    )

    np.testing.assert_allclose(east.delta_ht_db, 0.0)
    np.testing.assert_allclose(north.delta_ht_db, 20.0 * np.log10(0.25))
    np.testing.assert_allclose(
        vector.delta_ht_db,
        20.0 * np.log10(np.sqrt(5.0) / np.sqrt(20.0)),
    )
    assert east.ht_definition == "east"
    np.testing.assert_array_equal(east.ht_projection_east_north, (1.0, 0.0))
    assert np.all(np.isnan(vector.ht_projection_east_north))


def test_tangential_magnitude_and_vector_difference_are_not_conflated() -> None:
    time = np.linspace(0.0, 0.1, 11)
    reference = RadarTraces(
        time,
        np.ones_like(time),
        np.ones_like(time),
        np.ones_like(time),
        0.0,
        "reference",
        "test-signature",
    )
    anomaly = RadarTraces(
        time,
        np.ones_like(time),
        np.ones_like(time),
        -np.ones_like(time),
        0.0,
        "anomaly",
        "test-signature",
    )

    magnitude = compute_radar_perturbation(
        reference, anomaly, relative_stop_s=0.1, ht_definition="magnitude"
    )
    vector = compute_radar_perturbation(
        reference,
        anomaly,
        relative_stop_s=0.1,
        ht_definition="vector-difference",
    )

    assert np.max(magnitude.delta_ht_db) < -300.0
    np.testing.assert_allclose(vector.delta_ht_db, 20.0 * np.log10(np.sqrt(2.0)))


def test_radar_perturbation_rejects_unknown_tangential_definition() -> None:
    time = np.linspace(0.0, 0.1, 11)
    values = np.ones_like(time)
    reference = RadarTraces(
        time, values, values, values, 0.0, "reference", "test-signature"
    )
    anomaly = RadarTraces(
        time, values, values, values, 0.0, "anomaly", "test-signature"
    )

    with pytest.raises(ValueError, match="ht_definition"):
        compute_radar_perturbation(
            reference, anomaly, relative_stop_s=0.1, ht_definition="bearing"
        )


def test_radar_perturbation_rejects_incompatible_runs() -> None:
    time = np.linspace(0.0, 0.1, 11)
    values = np.sin(2.0 * np.pi * 20.0 * time)
    reference = RadarTraces(
        time, values, values, values, 0.0, "reference", "configuration-a"
    )
    incompatible = RadarTraces(
        time, values, values, values, 0.0, "anomaly", "configuration-b"
    )

    with pytest.raises(ValueError, match="signatures"):
        compute_radar_perturbation(reference, incompatible)


@requires_natural_earth
def test_recorded_radar_pair_has_matching_run_signature() -> None:
    reference_simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    anomaly_simulation = create_radar_simulation(
        include_oil=True,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    reference = record_radar_traces(
        reference_simulation, steps=1, case="reference"
    )
    anomaly = record_radar_traces(anomaly_simulation, steps=1, case="anomaly")

    assert reference.run_signature == anomaly.run_signature


@requires_natural_earth
def test_radar_trace_archive_preserves_run_signature(tmp_path) -> None:
    simulation = create_radar_simulation(
        include_oil=False,
        subdivision=0,
        material_model="natural-earth",
        dtype="float64",
        compile_step=False,
    )
    traces = record_radar_traces(simulation, steps=1, case="reference")

    restored = load_radar_traces(save_radar_traces(traces, tmp_path / "trace.npz"))

    assert restored.run_signature == traces.run_signature
    assert restored.case == "reference"
    np.testing.assert_array_equal(restored.hr_a_m, traces.hr_a_m)


def test_radar_trace_save_returns_normalized_npz_path(tmp_path) -> None:
    values = np.zeros(2)
    traces = RadarTraces(
        np.asarray((-0.5, 0.5)),
        values,
        values,
        values,
        0.0,
        "reference",
        "test-signature",
    )

    output = save_radar_traces(traces, tmp_path / "trace")

    assert output == tmp_path / "trace.npz"
    assert output.is_file()

from datetime import datetime, timezone

import numpy as np
import pytest

from ionosphere_fdtd.mesh import build_geodesic_mesh
from verification.simpson_taflove_2004.model import (
    PAPER_DFT_SIZE,
    PAPER_EVALUATION_FREQUENCIES_HZ,
    PAPER_RECEIVERS,
    PAPER_SOURCE_CENTER_STEPS,
    PAPER_SOURCE_FULL_WIDTH_STEPS,
    PAPER_TIME_STEP_S,
    REPRESENTATIVE_IONOSPHERE_REFERENCE_HEIGHT_M,
    REPRESENTATIVE_IONOSPHERE_SCALE_HEIGHT_M,
    ValidationTraces,
    arrival_metrics,
    bannister_figure_8_guide,
    bannister_phase_velocity_fraction_c,
    compute_attenuation,
    compute_phase_velocity,
    create_validation_simulation,
    equatorial_path_diagnostic_regions,
    find_dft_truncations,
    record_validation_traces,
    source_distribution_metrics,
    trace_metrics,
    validation_metrics,
)
from verification.simpson_taflove_2004.report import (
    ValidationRunSummary,
    write_validation_report,
)
from verification.simpson_taflove_2004.__main__ import (
    _parser as verification_parser,
    _reproduction_command,
)


def test_paper_setup_uses_delta_t_pulse_parameters() -> None:
    simulation = create_validation_simulation(
        subdivision=1,
        material_model="uniform",
        dtype="float64",
        compile_step=False,
    )

    assert simulation.time_step_s == pytest.approx(PAPER_TIME_STEP_S)
    assert simulation.config.radial_cells == 40
    assert simulation.source is not None
    assert simulation.source.latitude_deg == 0.0
    assert simulation.source.longitude_deg == -47.0
    assert simulation.source.center_time_s == pytest.approx(
        PAPER_SOURCE_CENTER_STEPS * PAPER_TIME_STEP_S
    )
    assert simulation.source.vertical_element_length_m == pytest.approx(5_000.0)
    assert simulation.config.loss_integration == "trapezoidal"
    assert simulation.config.geometry_mode == "thin-shell"
    assert simulation.source.one_over_e_half_width_s == pytest.approx(
        0.5 * PAPER_SOURCE_FULL_WIDTH_STEPS * PAPER_TIME_STEP_S
    )
    assert simulation.material.ionosphere_reference_height_m == pytest.approx(
        REPRESENTATIVE_IONOSPHERE_REFERENCE_HEIGHT_M
    )
    assert simulation.material.ionosphere_scale_height_m == pytest.approx(
        REPRESENTATIVE_IONOSPHERE_SCALE_HEIGHT_M
    )
    assert simulation.config.mesh_orientation == "polar"
    pentagons = simulation.mesh.vertices[simulation.mesh.vertex_degree == 5]
    assert np.max(pentagons[:, 2]) == pytest.approx(1.0)
    assert np.min(pentagons[:, 2]) == pytest.approx(-1.0)
    source_metrics = source_distribution_metrics(simulation)
    assert source_metrics["source_requested_altitude_m"] == 2_500.0
    assert source_metrics["source_staggered_centroid_altitude_m"] == 2_500.0
    assert source_metrics["source_staggered_lower_plane_altitude_m"] == 0.0
    assert source_metrics["source_staggered_upper_plane_altitude_m"] == 5_000.0
    assert source_metrics["source_staggered_radial_support_planes"] == 2
    assert source_metrics["source_distribution_weight_sum"] == pytest.approx(1.0)


def test_reproduction_command_records_all_result_changing_windows() -> None:
    args = verification_parser().parse_args(
        (
            "--radial-support",
            "dual-cell",
            "--tangential-interface",
            "fractional",
            "--tangential-support",
            "edge-diamond",
            "--spectral-window",
            "cosine-tail",
        )
    )

    command = _reproduction_command(args)

    assert "--radial-support dual-cell" in command
    assert "--tangential-interface fractional" in command
    assert "--tangential-support edge-diamond" in command
    assert "--spectral-window cosine-tail" in command
    assert "--dtype float64" in command
    assert "--deep-lithosphere-resistivity-ohm-m 500" in command


def test_reproduction_command_adds_tensorboard_dependency() -> None:
    args = verification_parser().parse_args(
        ("--tensorboard-log-dir", "/tmp/physics-events")
    )

    command = _reproduction_command(args)

    assert "uv run --extra tensorboard --extra visualization" in command
    assert "--tensorboard-log-dir /tmp/physics-events" in command
    assert "--diagnostics-every 512" in command


def test_validation_setup_can_retain_native_orientation() -> None:
    simulation = create_validation_simulation(
        subdivision=1,
        material_model="uniform",
        dtype="float64",
        compile_step=False,
        mesh_orientation="native",
    )

    north = int(np.argmax(simulation.mesh.vertices[:, 2]))
    south = int(np.argmin(simulation.mesh.vertices[:, 2]))
    assert simulation.config.mesh_orientation == "native"
    assert simulation.mesh.vertex_degree[north] == 6
    assert simulation.mesh.vertex_degree[south] == 6


def test_validation_setup_supports_cfl_safe_radial_refinement() -> None:
    simulation = create_validation_simulation(
        subdivision=1,
        radial_cells=80,
        time_step_s=0.5 * PAPER_TIME_STEP_S,
        material_model="uniform",
        dtype="float64",
        compile_step=False,
    )

    assert simulation.config.radial_cells == 80
    np.testing.assert_allclose(simulation.radial_steps_m, 2_500.0)
    assert simulation.time_step_s == pytest.approx(1.5e-6)
    assert simulation.source.center_time_s == pytest.approx(
        PAPER_SOURCE_CENTER_STEPS * PAPER_TIME_STEP_S
    )


def test_equatorial_diagnostic_corridors_are_disjoint_and_symmetric() -> None:
    simulation = create_validation_simulation(
        subdivision=3,
        material_model="uniform",
        dtype="float64",
        compile_step=False,
    )

    regions = equatorial_path_diagnostic_regions(simulation)

    assert set(regions) == {"east", "west"}
    for field in ("er_weights", "edge_weights", "hr_weights"):
        east = getattr(regions["east"], field)
        west = getattr(regions["west"], field)
        assert np.count_nonzero(east) > 0
        assert np.count_nonzero(west) > 0
        assert not np.any(east * west)
        assert np.count_nonzero(east) == pytest.approx(
            np.count_nonzero(west), rel=0.12
        )


def test_validation_setup_accepts_external_mesh() -> None:
    mesh = build_geodesic_mesh(1, optimization_steps=1)
    simulation = create_validation_simulation(
        subdivision=1,
        material_model="uniform",
        dtype="float64",
        compile_step=False,
        mesh=mesh,
    )

    assert simulation.mesh is mesh


def test_receiver_longitudes_follow_east_and_west_quarter_arcs() -> None:
    assert [(receiver.label, receiver.longitude_deg) for receiver in PAPER_RECEIVERS] == [
        ("A", -2.0),
        ("A′", -92.0),
        ("B", 43.0),
        ("B′", -137.0),
    ]


def test_short_validation_record_has_four_interpolated_receivers() -> None:
    simulation = create_validation_simulation(
        subdivision=0,
        material_model="uniform",
        dtype="float64",
        compile_step=False,
    )
    traces = record_validation_traces(simulation, steps=3)

    assert traces.er_v_m.shape == (4, 4)
    assert traces.time_steps.tolist() == [0, 1, 2, 3]
    assert traces.labels == ("A", "A′", "B", "B′")


def test_attenuation_recovers_known_spectral_ratio() -> None:
    count = 4_000
    time_steps = np.arange(count, dtype=np.int64)
    time = time_steps * PAPER_TIME_STEP_S
    wave = np.zeros(count)
    wave[1_000:2_000] = -np.sin(np.linspace(0.0, np.pi, 1_000))
    wave[2_000:2_500] = 0.2 * np.sin(np.linspace(0.0, np.pi, 500))
    wave[2_500:] = -0.02 * (1.0 - np.exp(-np.arange(count - 2_500) / 200.0))
    values = np.column_stack((wave, wave, 0.5 * wave, 0.25 * wave))
    traces = ValidationTraces(
        time_steps=time_steps,
        time_s=time,
        er_v_m=values,
        labels=("A", "A′", "B", "B′"),
    )

    curves = compute_attenuation(traces)
    assert curves.dft_truncations == {
        "A": 2_501,
        "A′": 2_501,
        "B": 2_501,
        "B′": 2_501,
    }
    index = int(np.argmin(np.abs(curves.frequency_hz - 200.0)))
    assert curves.path_ab_db_per_mm[index] > 0.0
    assert curves.path_apbp_db_per_mm[index] > curves.path_ab_db_per_mm[index]

    metrics = trace_metrics(traces)
    assert metrics["A_negative_peak_step"] == int(np.argmin(wave))
    assert metrics["quarter_east_west_relative_rms"] == pytest.approx(0.0)


def test_cosine_tail_window_preserves_scaled_attenuation_ratio() -> None:
    count = 4_000
    time_steps = np.arange(count, dtype=np.int64)
    wave = np.sin(np.linspace(0.0, 8.0 * np.pi, count))
    traces = ValidationTraces(
        time_steps=time_steps,
        time_s=time_steps * PAPER_TIME_STEP_S,
        er_v_m=np.column_stack((wave, wave, 0.5 * wave, 0.25 * wave)),
        labels=("A", "A′", "B", "B′"),
    )
    truncations = dict.fromkeys(traces.labels, count)

    rectangular = compute_attenuation(traces, truncations=truncations)
    tapered = compute_attenuation(
        traces,
        truncations=truncations,
        spectral_window="cosine-tail",
    )

    np.testing.assert_allclose(
        tapered.path_ab_db_per_mm, rectangular.path_ab_db_per_mm
    )
    np.testing.assert_allclose(
        tapered.path_apbp_db_per_mm, rectangular.path_apbp_db_per_mm
    )


def test_attenuation_rejects_unknown_spectral_window() -> None:
    time_steps = np.arange(8, dtype=np.int64)
    traces = ValidationTraces(
        time_steps=time_steps,
        time_s=time_steps * PAPER_TIME_STEP_S,
        er_v_m=np.ones((8, 4)),
        labels=("A", "A′", "B", "B′"),
    )

    with pytest.raises(ValueError, match="spectral_window"):
        compute_attenuation(
            traces,
            truncations=dict.fromkeys(traces.labels, 8),
            spectral_window="unknown",
        )


def test_bannister_guide_matches_published_daytime_examples() -> None:
    attenuation = bannister_figure_8_guide(np.asarray([0.0, 75.0, 1_000.0]))

    assert attenuation[0] == 0.0
    assert attenuation[1] == pytest.approx(1.5, rel=0.01)
    assert attenuation[2] == pytest.approx(16.6, rel=0.01)


def test_bannister_phase_velocity_matches_published_daytime_examples() -> None:
    velocity = bannister_phase_velocity_fraction_c(np.asarray([75.0, 1_000.0]))

    assert velocity[0] == pytest.approx(1.0 / 1.26, rel=0.01)
    assert velocity[1] == pytest.approx(1.0 / 1.10, rel=0.01)


def test_phase_velocity_recovers_known_receiver_delay() -> None:
    count = 25_024
    time_steps = np.arange(count, dtype=np.int64)
    near = -np.exp(-((time_steps - 3_000) / 300.0) ** 2)
    far = -np.exp(-((time_steps - 9_000) / 300.0) ** 2)
    traces = ValidationTraces(
        time_steps=time_steps,
        time_s=time_steps * PAPER_TIME_STEP_S,
        er_v_m=np.column_stack((near, near, far, far)),
        labels=("A", "A′", "B", "B′"),
    )
    truncations = dict.fromkeys(traces.labels, count)

    curves = compute_phase_velocity(traces, truncations=truncations)
    padded = compute_phase_velocity(
        traces,
        n_fft=2 * PAPER_DFT_SIZE,
        truncations=truncations,
    )
    metrics = arrival_metrics(traces)
    expected = (
        0.25
        * np.pi
        * 6_371_000.0
        / (6_000 * PAPER_TIME_STEP_S)
        / 299_792_458.0
    )

    np.testing.assert_allclose(curves.path_ab_fraction_c, expected, rtol=1.0e-13)
    np.testing.assert_allclose(curves.path_apbp_fraction_c, expected, rtol=1.0e-13)
    np.testing.assert_allclose(
        padded.path_ab_fraction_c,
        curves.path_ab_fraction_c,
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        padded.path_apbp_fraction_c,
        curves.path_apbp_fraction_c,
        rtol=1.0e-13,
    )
    assert metrics["path_ab_apparent_peak_velocity_fraction_c"] == pytest.approx(
        expected, rel=1.0e-12
    )


def test_validation_uses_fixed_paper_dft_frequencies() -> None:
    frequencies = np.asarray(PAPER_EVALUATION_FREQUENCIES_HZ)

    assert len(frequencies) == 45
    assert frequencies[0] == pytest.approx(50.86263020833333)
    assert frequencies[-1] == pytest.approx(498.45377604166663)

    count = 25_024
    time_steps = np.arange(count, dtype=np.int64)
    wave = np.zeros(count)
    wave[7_000:9_000] = -np.sin(np.linspace(0.0, np.pi, 2_000))
    wave[9_000:10_000] = 0.2 * np.sin(np.linspace(0.0, np.pi, 1_000))
    wave[10_000:] = -0.01 * (
        1.0 - np.exp(-np.arange(count - 10_000) / 500.0)
    )
    traces = ValidationTraces(
        time_steps=time_steps,
        time_s=time_steps * PAPER_TIME_STEP_S,
        er_v_m=np.column_stack((wave, wave, 0.5 * wave, 0.25 * wave)),
        labels=("A", "A′", "B", "B′"),
    )

    base = validation_metrics(compute_attenuation(traces, n_fft=PAPER_DFT_SIZE))
    padded = validation_metrics(
        compute_attenuation(traces, n_fft=2 * PAPER_DFT_SIZE)
    )
    assert padded == pytest.approx(base, rel=1.0e-12, abs=1.0e-12)


def test_adaptive_dft_window_rejects_trace_without_positive_overshoot() -> None:
    time_steps = np.arange(4_000, dtype=np.int64)
    wave = -np.exp(-((time_steps - 2_000) / 300.0) ** 2)
    traces = ValidationTraces(
        time_steps=time_steps,
        time_s=time_steps * PAPER_TIME_STEP_S,
        er_v_m=np.column_stack((wave, wave, wave, wave)),
        labels=("A", "A′", "B", "B′"),
    )

    with pytest.raises(ValueError, match="no positive overshoot"):
        find_dft_truncations(traces)


def test_markdown_report_records_configuration_results_and_artifacts(tmp_path) -> None:
    figure_7 = tmp_path / "fig-7.png"
    figure_8 = tmp_path / "fig-8.png"
    summary = ValidationRunSummary(
        generated_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
        command=(
            "uv run python -m verification.simpson_taflove_2004 "
            "--subdivision 7"
        ),
        git_revision="abc1234",
        subdivision=7,
        mesh_optimization_steps=0,
        minimum_ocean_depth_m=0.0,
        deep_lithosphere_resistivity_ohm_m=500.0,
        surface_cells=163_842,
        radial_cells=40,
        time_step_s=3.0e-6,
        steps=35_000,
        material_model="natural-earth",
        relief_data=None,
        ionosphere_reference_height_m=70_000.0,
        ionosphere_scale_height_m=1_000.0 / 0.3,
        dft_window="adaptive",
        spectral_window="rectangular",
        radial_support="point",
        tangential_interface="point",
        tangential_support="point",
        runtime="torch",
        device="mps",
        dtype="float32",
        compiled=True,
        elapsed_s=609.5,
        metrics={
            "path_ab_mean_absolute_error_db_per_mm": 6.146,
            "path_apbp_mean_absolute_error_db_per_mm": 5.991,
            "path_ab_maximum_absolute_error_db_per_mm": 9.0,
            "path_apbp_maximum_absolute_error_db_per_mm": 8.0,
            "A_negative_peak_step": 8_800,
        },
        figure_7=figure_7,
        figure_8=figure_8,
        trace_data=tmp_path / "traces.npz",
    )

    report = write_validation_report(summary, tmp_path / "report.md")
    text = report.read_text(encoding="utf-8")

    assert "정량 검증 상태: **실패**" in text
    assert "mesh optimization steps | 0" in text
    assert "minimum ocean depth | 0 km" in text
    assert "deep lithosphere resistivity | 500 Ω·m" in text
    assert "163,842" in text
    assert "6.146 dB/Mm" in text
    assert "9.000 dB/Mm" in text
    assert "`adaptive`" in text
    assert "spectral window | `rectangular`" in text
    assert "radial support | `point`" in text
    assert "45개 bin" in text
    assert "Bannister (1984)" in text
    assert "daytime phase velocity" in text
    assert "Natural Earth 110-m" in text
    assert "![Figure 7 verification](fig-7.png)" in text
    assert "[Receiver traces (NPZ)](traces.npz)" in text
    assert "uv run python -m verification.simpson_taflove_2004" in text

from pathlib import Path

import numpy as np
import pytest
import torch

from ionosphere_fdtd.constants import EPSILON_0, MU_0
from verification.physics_diagnostics import (
    HorizontalRegion,
    PhysicsDiagnosticSampler,
    TensorBoardPhysicsRecorder,
    save_physics_snapshots,
)
from verification.simpson_taflove_2004.model import (
    create_validation_simulation,
    record_validation_traces,
)


class _SnapshotRecorder:
    def __init__(self, simulation) -> None:
        self.sampler = PhysicsDiagnosticSampler(simulation)
        self.snapshots = []

    def record(self, receiver_values, *, steps_per_second=None):
        snapshot = self.sampler.sample(
            receiver_values, steps_per_second=steps_per_second
        )
        self.snapshots.append(snapshot)
        return snapshot


def _small_simulation(*, device: str = "cpu"):
    return create_validation_simulation(
        subdivision=1,
        material_model="uniform",
        device=device,
        dtype="float64",
        compile_step=False,
    )


def test_physics_sampler_matches_independent_volume_integrals() -> None:
    simulation = _small_simulation()
    simulation.er.fill_(2.0)
    simulation.et.fill_(3.0)
    simulation.hr.fill_(4.0)
    simulation.ht.fill_(5.0)
    fields_before = tuple(
        values.clone()
        for values in (simulation.er, simulation.et, simulation.hr, simulation.ht)
    )

    snapshot = PhysicsDiagnosticSampler(simulation).sample({"A": 1.25})

    node_volume = (
        simulation.mesh.dual_cell_solid_angles[:, None]
        * (
            simulation.radii_m**2
            * simulation.radial_node_control_lengths_m
        )[None, :]
    )
    edge_node_volume = (
        (
            simulation.mesh.primal_edge_angles
            * simulation.mesh.dual_edge_angles
        )[:, None]
        * (
            simulation.radii_m**2
            * simulation.radial_node_control_lengths_m
        )[None, :]
    )
    edge_cell_volume = (
        (
            simulation.mesh.primal_edge_angles
            * simulation.mesh.dual_edge_angles
        )[:, None]
        * (
            simulation.radial_midpoints_m**2 * simulation.radial_steps_m
        )[None, :]
    )
    face_cell_volume = (
        simulation.mesh.face_solid_angles[:, None]
        * (
            simulation.radial_midpoints_m**2 * simulation.radial_steps_m
        )[None, :]
    )
    expected_er = 0.5 * np.sum(
        EPSILON_0 * simulation.to_numpy(simulation.epsilon_r_er) * 2.0**2 * node_volume
    )
    expected_et = 0.5 * np.sum(
        EPSILON_0 * simulation.to_numpy(simulation.epsilon_r_et) * 3.0**2 * edge_cell_volume
    )
    expected_hr = 0.5 * MU_0 * 4.0**2 * np.sum(face_cell_volume)
    expected_ht = 0.5 * MU_0 * 5.0**2 * np.sum(edge_node_volume)
    expected_loss = np.sum(simulation.to_numpy(simulation.sigma_er) * 2.0**2 * node_volume)
    expected_loss += np.sum(simulation.to_numpy(simulation.sigma_et) * 3.0**2 * edge_cell_volume)

    assert snapshot.scalars["energy/er_j"] == pytest.approx(expected_er)
    assert snapshot.scalars["energy/et_j"] == pytest.approx(expected_et)
    assert snapshot.scalars["energy/hr_j"] == pytest.approx(expected_hr)
    assert snapshot.scalars["energy/ht_j"] == pytest.approx(expected_ht)
    assert snapshot.scalars["conductive_loss/total_w"] == pytest.approx(
        expected_loss
    )
    assert snapshot.receiver_values == {"A": 1.25}
    assert all(
        snapshot.scalars[f"field_finite/{name}"] == 1.0
        for name in ("er", "et", "hr", "ht")
    )
    for before, after in zip(
        fields_before,
        (simulation.er, simulation.et, simulation.hr, simulation.ht),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)


def test_chunked_diagnostics_do_not_change_traces_or_fields(
    tmp_path: Path,
) -> None:
    direct = _small_simulation()
    observed = _small_simulation()
    expected = record_validation_traces(direct, steps=12)
    recorder = _SnapshotRecorder(observed)

    actual = record_validation_traces(
        observed,
        steps=12,
        diagnostics_every=5,
        recorder=recorder,
    )

    np.testing.assert_array_equal(actual.time_steps, expected.time_steps)
    np.testing.assert_array_equal(actual.time_s, expected.time_s)
    np.testing.assert_array_equal(actual.er_v_m, expected.er_v_m)
    for direct_field, observed_field in zip(
        (direct.er, direct.et, direct.hr, direct.ht),
        (observed.er, observed.et, observed.hr, observed.ht),
        strict=True,
    ):
        np.testing.assert_array_equal(observed_field, direct_field)
    assert [item.step for item in recorder.snapshots] == [0, 5, 10, 12]
    assert all(
        tuple(sorted(item.scalars))
        == tuple(sorted(recorder.snapshots[0].scalars))
        for item in recorder.snapshots
    )
    assert save_physics_snapshots(
        tmp_path / "chunked.npz",
        recorder.snapshots,
        node_altitudes_m=observed.altitudes_m,
        cell_altitudes_m=observed.radial_midpoint_altitudes_m,
    ).is_file()


def test_full_horizontal_region_matches_global_energy_and_loss() -> None:
    simulation = _small_simulation()
    simulation.er.fill_(2.0)
    simulation.et.fill_(3.0)
    simulation.hr.fill_(4.0)
    simulation.ht.fill_(5.0)
    full = HorizontalRegion(
        np.ones(simulation.mesh.n_vertices),
        np.ones(simulation.mesh.n_edges),
        np.ones(simulation.mesh.n_faces),
    )

    snapshot = PhysicsDiagnosticSampler(
        simulation, horizontal_regions={"full": full}
    ).sample()

    assert snapshot.scalars["energy_horizontal_region/full_j"] == pytest.approx(
        snapshot.scalars["energy/total_staggered_j"]
    )
    assert snapshot.scalars[
        "conductive_loss_horizontal_region/full_w"
    ] == pytest.approx(snapshot.scalars["conductive_loss/total_w"])
    regional_loss = sum(
        snapshot.scalars[
            f"conductive_loss_horizontal_region/full/{region}_w"
        ]
        for region in ("earth", "atmosphere", "ionosphere")
    )
    assert regional_loss == pytest.approx(
        snapshot.scalars["conductive_loss/total_w"]
    )


def test_cuda_sampler_matches_numpy_float64_reductions() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu = _small_simulation()
    cuda = _small_simulation(device="cuda:0")
    for values, constant in zip(
        (cpu.er, cpu.et, cpu.hr, cpu.ht),
        (2.0, 3.0, 4.0, 5.0),
        strict=True,
    ):
        values.fill_(constant)
    for values, constant in zip(
        (cuda.er, cuda.et, cuda.hr, cuda.ht),
        (2.0, 3.0, 4.0, 5.0),
        strict=True,
    ):
        values.fill_(constant)

    cpu_snapshot = PhysicsDiagnosticSampler(cpu).sample()
    cuda_snapshot = PhysicsDiagnosticSampler(cuda).sample()

    for name in (
        "energy/er_j",
        "energy/et_j",
        "energy/hr_j",
        "energy/ht_j",
        "conductive_loss/total_w",
    ):
        assert cuda_snapshot.scalars[name] == pytest.approx(
            cpu_snapshot.scalars[name], rel=2.0e-15
        )


def test_tensorboard_events_and_npz_preserve_physics_samples(
    tmp_path: Path,
) -> None:
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    simulation = _small_simulation()
    recorder = TensorBoardPhysicsRecorder(
        simulation,
        tmp_path / "events",
        metadata={"case": "unit-test"},
    )
    recorder.record({"A": 0.0})
    simulation.step(2)
    recorder.record({"A": simulation.field_value("er", 0, 0)})
    recorder.close()

    output = save_physics_snapshots(
        tmp_path / "physics.npz",
        recorder.snapshots,
        node_altitudes_m=simulation.altitudes_m,
        cell_altitudes_m=simulation.radial_midpoint_altitudes_m,
        metadata=recorder.metadata,
    )
    events = event_accumulator.EventAccumulator(str(tmp_path / "events"))
    events.Reload()

    assert "energy/total_staggered_j" in events.Tags()["scalars"]
    assert "receiver/A_v_m" in events.Tags()["scalars"]
    assert len(events.Scalars("energy/total_staggered_j")) == 2
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["steps"], (0, 2))
        assert archive["radial_energy_er_j"].shape == (2, 41)
        assert archive["radial_energy_et_j"].shape == (2, 40)
        assert archive["receiver_labels"].tolist() == ["A"]

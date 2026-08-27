import json

import numpy as np
import pytest

from ionosphere_fdtd import CheckpointError
from ionosphere_fdtd.cli import main
from ionosphere_fdtd.materials import (
    EarthIonosphereMaterial,
    LayeredEarthIonosphereMaterial,
    SphericalAnomaly,
)
from ionosphere_fdtd.mesh import (
    build_geodesic_mesh,
    build_geodesic_mesh_from_topology,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent, TangentialGaussianCurrent
from ionosphere_fdtd.surface_impedance import ConductiveHalfSpaceSurface


def checkpoint_config(**changes: object) -> SimulationConfig:
    values = dict(subdivision=1, radial_cells=6, courant_factor=0.25)
    values.update(changes)
    return SimulationConfig(**values)


def assert_same_state(first: GeodesicFDTD, second: GeodesicFDTD) -> None:
    assert second.config == first.config
    assert second.material == first.material
    assert second.source == first.source
    if first.surface_impedance is None:
        assert second.surface_impedance is None
        assert second._surface_impedance_ade is None
    else:
        assert second.surface_impedance is not None
        assert (
            second.surface_impedance.to_metadata()
            == first.surface_impedance.to_metadata()
        )
        np.testing.assert_array_equal(
            second.to_numpy(second._surface_impedance_ade.memory),
            first.to_numpy(first._surface_impedance_ade.memory),
        )
    if first.plasma is None:
        assert second.plasma is None
        assert second._plasma_coupler is None
    else:
        assert second.plasma is not None
        assert second.plasma.content_sha256 == first.plasma.content_sha256
        for second_current, first_current in zip(
            second._plasma_coupler.ade.current_density,
            first._plasma_coupler.ade.current_density,
            strict=True,
        ):
            np.testing.assert_array_equal(
                second.to_numpy(second_current), first.to_numpy(first_current)
            )
    assert second.steps == first.steps
    assert second.time_s == pytest.approx(first.time_s)
    np.testing.assert_array_equal(second.mesh.vertices, first.mesh.vertices)
    np.testing.assert_array_equal(second.mesh.faces, first.mesh.faces)
    assert second.mesh.topology_kind == first.mesh.topology_kind
    assert second.mesh.subdivision == first.mesh.subdivision
    assert second.mesh.refinement_spec == first.mesh.refinement_spec
    if first.mesh.face_levels is None:
        assert second.mesh.face_levels is None
    else:
        np.testing.assert_array_equal(
            second.mesh.face_levels, first.mesh.face_levels
        )
    for name in ("er", "et", "hr", "ht"):
        np.testing.assert_array_equal(
            second.to_numpy(getattr(second, name)),
            first.to_numpy(getattr(first, name)),
        )


def test_checkpoint_round_trip_and_continuation(tmp_path) -> None:
    material = EarthIonosphereMaterial(
        anomalies=(
            SphericalAnomaly(35.0, 126.0, 500_000.0, -20_000.0, -1_000.0, 0.5),
        )
    )
    source = GaussianCurrent(
        peak_current_a=2.0e6,
        center_time_s=0.01,
        one_over_e_half_width_s=0.02,
        carrier_frequency_hz=5.0,
    )
    uninterrupted = GeodesicFDTD(checkpoint_config(), material, source)
    resumed_source = GeodesicFDTD(checkpoint_config(), material, source)
    uninterrupted.step(12)
    resumed_source.step(5)

    path = resumed_source.save_checkpoint(tmp_path / "state.npz")
    resumed = GeodesicFDTD.load_checkpoint(path)
    assert_same_state(resumed_source, resumed)

    resumed.step(7)
    assert_same_state(uninterrupted, resumed)


def test_checkpoint_preserves_tangential_source_and_optimized_mesh(tmp_path) -> None:
    source = TangentialGaussianCurrent(
        azimuths_deg=(15.0, 90.0),
        line_lengths_m=(10_000.0, 20_000.0),
        edge_assignment="nearest",
    )
    simulation = GeodesicFDTD(
        checkpoint_config(mesh_optimization_steps=1), source=source, dtype="float32"
    )
    simulation.step(3)

    restored = GeodesicFDTD.load_checkpoint(
        simulation.save_checkpoint(tmp_path / "tangential.npz")
    )

    assert restored.dtype_name == "float32"
    assert_same_state(simulation, restored)

    converted = GeodesicFDTD.load_checkpoint(
        tmp_path / "tangential.npz", dtype="float64"
    )
    assert converted.dtype_name == "float64"
    for name in ("er", "et", "hr", "ht"):
        np.testing.assert_array_equal(
            converted.to_numpy(getattr(converted, name)),
            simulation.to_numpy(getattr(simulation, name)).astype(np.float64),
        )


def test_checkpoint_v4_preserves_custom_topology_and_refinement_metadata(
    tmp_path,
) -> None:
    uniform = build_geodesic_mesh(1)
    faces = np.roll(uniform.faces, 7, axis=0)
    face_levels = np.where(np.arange(len(faces)) % 2, 2, 1)
    mesh = build_geodesic_mesh_from_topology(
        uniform.vertices,
        faces,
        subdivision=1,
        face_levels=face_levels,
        refinement_spec={"balance": "2:1", "regions": [{"level": 2}]},
    )
    simulation = GeodesicFDTD(checkpoint_config(), mesh=mesh)
    simulation.step(3)

    path = simulation.save_checkpoint(tmp_path / "adaptive.npz")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["version"] == 4
        np.testing.assert_array_equal(archive["mesh_faces"], faces)
        np.testing.assert_array_equal(archive["mesh_face_levels"], face_levels)

    restored = GeodesicFDTD.load_checkpoint(path)
    assert_same_state(simulation, restored)
    simulation.step(2)
    restored.step(2)
    assert_same_state(simulation, restored)


def test_checkpoint_loads_legacy_v1_uniform_mesh(tmp_path) -> None:
    simulation = GeodesicFDTD(checkpoint_config())
    path = simulation.save_checkpoint(tmp_path / "current.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name not in {"mesh_faces", "mesh_face_levels"}
        }
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["version"] = 1
    metadata["runtime"] = {"backend": "numpy", "dtype": "float64"}
    metadata.pop("mesh")
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    legacy = tmp_path / "legacy-v1.npz"
    np.savez_compressed(legacy, **arrays)

    restored = GeodesicFDTD.load_checkpoint(legacy, device="cpu", dtype="float32")

    assert restored.runtime == "torch"
    assert restored.dtype_name == "float32"
    assert_same_state(simulation, restored)


def test_checkpoint_loads_legacy_v2_without_surface_state(tmp_path) -> None:
    simulation = GeodesicFDTD(checkpoint_config())
    path = simulation.save_checkpoint(tmp_path / "current.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "surface_impedance_memory"
        }
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["version"] = 2
    metadata["runtime"] = {"backend": "numpy", "dtype": "float64"}
    metadata.pop("surface_impedance")
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    legacy = tmp_path / "legacy-v2.npz"
    np.savez_compressed(legacy, **arrays)

    restored = GeodesicFDTD.load_checkpoint(legacy, device="cpu", dtype="float32")

    assert restored.runtime == "torch"
    assert restored.dtype_name == "float32"
    assert_same_state(simulation, restored)


def test_checkpoint_loads_legacy_v3_on_selected_runtime(tmp_path) -> None:
    simulation = GeodesicFDTD(checkpoint_config())
    current = simulation.save_checkpoint(tmp_path / "current-v4.npz")
    with np.load(current, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["version"] = 3
    metadata["runtime"] = {"backend": "numpy", "dtype": "float64"}
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    legacy = tmp_path / "legacy-v3.npz"
    np.savez_compressed(legacy, **arrays)

    restored = GeodesicFDTD.load_checkpoint(legacy, device="cpu", dtype="float32")

    assert restored.runtime == "torch"
    assert restored.dtype_name == "float32"
    assert_same_state(simulation, restored)


def test_checkpoint_preserves_surface_impedance_ade_state(tmp_path) -> None:
    config = SimulationConfig(
        subdivision=0,
        radial_cells=4,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
        radial_boundary_condition="surface-impedance",
    )
    surface = ConductiveHalfSpaceSurface(
        np.linspace(0.01, 0.03, build_geodesic_mesh(0).n_edges)
    )
    source = GaussianCurrent(peak_current_a=1.0e6)
    uninterrupted = GeodesicFDTD(
        config, source=source, surface_impedance=surface
    )
    checkpointed = GeodesicFDTD(
        config, source=source, surface_impedance=surface
    )
    uninterrupted.step(12)
    checkpointed.step(5)

    restored = GeodesicFDTD.load_checkpoint(
        checkpointed.save_checkpoint(tmp_path / "surface.npz")
    )
    assert_same_state(checkpointed, restored)
    restored.step(7)
    assert_same_state(uninterrupted, restored)


def test_checkpoint_rejects_unsupported_material(tmp_path) -> None:
    material = LayeredEarthIonosphereMaterial(
        land_classifier=lambda directions: np.ones(len(directions), dtype=np.bool_)
    )
    simulation = GeodesicFDTD(checkpoint_config(), material=material)

    with pytest.raises(CheckpointError, match="EarthIonosphereMaterial only"):
        simulation.save_checkpoint(tmp_path / "unsupported.npz")


def test_checkpoint_rejects_wrong_version(tmp_path) -> None:
    simulation = GeodesicFDTD(checkpoint_config())
    original = simulation.save_checkpoint(tmp_path / "original.npz")
    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["version"] = 999
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, **arrays)

    with pytest.raises(CheckpointError, match="unsupported checkpoint version"):
        GeodesicFDTD.load_checkpoint(invalid)


def test_cli_writes_and_resumes_checkpoint(tmp_path, capsys) -> None:
    checkpoint = tmp_path / "cli-state.npz"
    assert main(
        [
            "--subdivision",
            "0",
            "--radial-cells",
            "4",
            "--steps",
            "3",
            "--report-every",
            "2",
            "--checkpoint-every",
            "2",
            "--checkpoint",
            str(checkpoint),
        ]
    ) == 0
    assert checkpoint.exists()
    assert GeodesicFDTD.load_checkpoint(checkpoint).steps == 3

    assert main(["--resume", str(checkpoint), "--steps", "2"]) == 0
    output = capsys.readouterr().out
    assert "step=     5" in output

import numpy as np
import pytest
import torch

from ionosphere_fdtd.data_artifacts import (
    DatasetProvenance,
    VariableProvenance,
)
from ionosphere_fdtd.mesh import build_geodesic_mesh
from ionosphere_fdtd.plasma import (
    ColdPlasmaSpecies,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    MagnetizedPlasmaADE,
    MeshPlasmaModel,
    _face_reconstruction,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig


class _ArrayRuntime:
    def as_tensor(self, values):
        return np.asarray(values, dtype=np.float64)

    def index_tensor(self, values):
        return np.asarray(values, dtype=np.int64)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def empty_like(self, values):
        return np.empty_like(values)

    def nbytes(self, values):
        return values.nbytes


def _provenance() -> DatasetProvenance:
    return DatasetProvenance(
        dataset_id="test.plasma.synthetic.v1",
        title="Synthetic cold-plasma regression data",
        version="1",
        source_url="https://example.invalid/plasma-v1",
        citation="Synthetic fixture.",
        license="CC0-1.0",
        retrieved_at="2026-08-20T10:00:00Z",
        source_sha256="0" * 64,
        coordinate_reference_system="geocentric Cartesian, WGS 84 altitude",
        variables=(
            VariableProvenance("B", "T", "T", "identity"),
            VariableProvenance("number_density", "m^-3", "m^-3", "identity"),
            VariableProvenance("collision_frequency", "Hz", "Hz", "identity"),
        ),
    )


def _model(
    *,
    magnetic_t: float = 50.0e-6,
    density_m3: float = 1.0e3,
    collision_hz: float = 200.0,
    mesh=None,
) -> tuple[MeshPlasmaModel, np.ndarray]:
    mesh = build_geodesic_mesh(0) if mesh is None else mesh
    altitudes = np.asarray((25_000.0, 75_000.0))
    shape = (mesh.n_faces, len(altitudes))
    magnetic = np.zeros((*shape, 3))
    magnetic[..., 2] = magnetic_t
    species = ColdPlasmaSpecies(
        "electron",
        -ELEMENTARY_CHARGE_C,
        ELECTRON_MASS_KG,
        np.full(shape, density_m3),
        np.full(shape, collision_hz),
    )
    return (
        MeshPlasmaModel.from_mesh(
            mesh,
            altitudes,
            magnetic,
            (species,),
            provenance=(_provenance(),),
            interpolation="synthetic constants at face/cell centers",
        ),
        altitudes,
    )


def test_unmagnetized_conductivity_is_isotropic() -> None:
    model, _ = _model(magnetic_t=0.0)

    parallel, pedersen, hall = model.conductivity_components(20.0)

    np.testing.assert_allclose(pedersen, parallel, rtol=3.0e-16, atol=0.0)
    np.testing.assert_array_equal(hall, 0.0)
    assert np.all(parallel.real > 0.0)


def test_face_reconstruction_recovers_tangent_vectors() -> None:
    mesh = build_geodesic_mesh(1)
    reconstruction, edge_tangents = _face_reconstruction(mesh)
    vector = np.asarray((0.3, -0.4, 0.5))
    tangent_vector = vector - (mesh.face_centers @ vector)[:, None] * (
        mesh.face_centers
    )
    samples = np.sum(edge_tangents * tangent_vector[:, None, :], axis=2)
    recovered = np.sum(samples[:, :, None] * reconstruction, axis=1)

    np.testing.assert_allclose(recovered, tangent_vector, rtol=0.0, atol=8.0e-16)


def test_magnetized_tensor_has_passive_pedersen_and_signed_hall_terms() -> None:
    model, _ = _model()

    parallel, pedersen, hall = model.conductivity_components(20.0)

    assert np.all(parallel.real > 0.0)
    assert np.all(pedersen.real > 0.0)
    assert np.all(hall.real < 0.0)
    assert np.max(np.abs(hall)) > np.max(np.abs(pedersen))


def test_vector_ade_matches_complex_tensor_response() -> None:
    model, _ = _model(magnetic_t=1.0e-8, density_m3=1.0e6)
    one_cell = MeshPlasmaModel(
        mesh_vertices_sha256=model.mesh_vertices_sha256,
        mesh_faces_sha256=model.mesh_faces_sha256,
        radial_midpoint_altitudes_m=np.asarray((25_000.0,)),
        magnetic_field_t=model.magnetic_field_t[:1, :1],
        species=(
            ColdPlasmaSpecies(
                "electron",
                -ELEMENTARY_CHARGE_C,
                ELECTRON_MASS_KG,
                model.species[0].number_density_m3[:1, :1],
                model.species[0].collision_frequency_hz[:1, :1],
            ),
        ),
        provenance=model.provenance,
        interpolation=model.interpolation,
    )
    time_step = 1.0e-5
    frequency = 20.0
    ade = MagnetizedPlasmaADE(one_cell, time_step, _ArrayRuntime())
    samples = 100_000
    electric = np.zeros((1, 1, 3))
    current = np.empty((samples, 2))
    drive = np.empty(samples)
    for step in range(samples):
        time = step * time_step
        drive[step] = np.cos(2.0 * np.pi * frequency * time)
        electric[..., 0] = drive[step]
        current[step] = ade.advance(electric)[0, 0, :2]

    selected = slice(samples - 20_000, None)
    time = np.arange(samples)[selected] * time_step
    basis = np.exp(-2j * np.pi * frequency * (time + time_step))
    denominator = np.sum(drive[selected] * basis)
    measured_x = np.sum(current[selected, 0] * basis) / denominator
    measured_y = np.sum(current[selected, 1] * basis) / denominator
    _, pedersen, hall = one_cell.conductivity_components(frequency)
    assert measured_x == pytest.approx(pedersen[0, 0], rel=2.0e-3)
    assert measured_y == pytest.approx(-hall[0, 0], rel=2.0e-3)


def test_solver_couples_vector_plasma_current_and_limits_time_step() -> None:
    mesh = build_geodesic_mesh(0)
    model, _ = _model(density_m3=1.0e8, mesh=mesh)
    config = SimulationConfig(
        subdivision=0,
        radial_cells=2,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
    )
    plasma = GeodesicFDTD(config, mesh=mesh, plasma=model)
    vacuum = GeodesicFDTD(config, mesh=mesh)
    values = np.random.default_rng(20260820).standard_normal(plasma.et.shape) * 1.0e-6
    plasma.et.copy_(torch.as_tensor(values))
    vacuum.et.copy_(torch.as_tensor(values))

    plasma._update_electric_fields()
    vacuum._update_electric_fields()

    assert torch.max(torch.abs(plasma.et - vacuum.et)).item() > 0.0
    assert plasma._plasma_coupler.ade.state_bytes > 0
    assert plasma.cfl_time_step_limit_s < vacuum.cfl_time_step_limit_s
    assert plasma.diagnostics()["plasma_species"] == 1


def test_plasma_model_rejects_different_mesh() -> None:
    model, altitudes = _model()

    with pytest.raises(ValueError, match="vertices do not match"):
        model.validate_grid(build_geodesic_mesh(1), altitudes)


def test_mesh_plasma_artifact_round_trip_and_corruption_detection(tmp_path) -> None:
    model, _ = _model()
    path = model.save(tmp_path / "plasma.npz")

    restored = MeshPlasmaModel.load(path)

    assert restored.content_sha256 == model.content_sha256
    assert not restored.magnetic_field_t.flags.writeable
    np.testing.assert_array_equal(
        restored.species[0].number_density_m3,
        model.species[0].number_density_m3,
    )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["magnetic_field_t"][0, 0, 0] += 1.0e-9
    corrupt = tmp_path / "corrupt.npz"
    np.savez_compressed(corrupt, **arrays)
    with pytest.raises(ValueError, match="magnetic_field_t checksum mismatch"):
        MeshPlasmaModel.load(corrupt)


def test_torch_compiled_plasma_matches_eager_and_checkpoint_resumes(
    tmp_path,
) -> None:
    mesh = build_geodesic_mesh(0)
    model, _ = _model(density_m3=1.0e4, mesh=mesh)
    config = SimulationConfig(
        subdivision=0,
        radial_cells=2,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
    )
    eager = GeodesicFDTD(
        config,
        mesh=mesh,
        plasma=model,
        device="cpu",
        dtype="float64",
    )
    compiled = GeodesicFDTD(
        config,
        mesh=mesh,
        plasma=model,
        device="cpu",
        dtype="float64",
        compile_step=True,
        compile_chunk_size=1,
    )
    generator = np.random.default_rng(20260820)
    for name in ("er", "et", "hr", "ht"):
        values = generator.standard_normal(getattr(eager, name).shape) * 1.0e-9
        getattr(eager, name).copy_(eager._runtime.as_tensor(values))
        getattr(compiled, name).copy_(compiled._runtime.as_tensor(values))

    eager.step(2)
    compiled.step(2)

    for name in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            compiled.to_numpy(getattr(compiled, name)),
            eager.to_numpy(getattr(eager, name)),
            rtol=3.0e-13,
            atol=1.0e-22,
        )
    restored = GeodesicFDTD.load_checkpoint(
        eager.save_checkpoint(tmp_path / "plasma-state.npz"),
        device="cpu",
        dtype="float64",
    )
    assert restored.plasma.content_sha256 == model.content_sha256
    for restored_current, eager_current in zip(
        restored._plasma_coupler.ade.current_density,
        eager._plasma_coupler.ade.current_density,
        strict=True,
    ):
        np.testing.assert_array_equal(
            restored.to_numpy(restored_current), eager.to_numpy(eager_current)
        )
    eager.step()
    restored.step()
    for name in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            restored.to_numpy(getattr(restored, name)),
            eager.to_numpy(getattr(eager, name)),
            rtol=3.0e-13,
            atol=1.0e-22,
        )

import numpy as np
import pytest

from ionosphere_fdtd.backends import BackendUnavailableError
from ionosphere_fdtd.backends.numpy_backend import NumPyBackend
from ionosphere_fdtd.mesh import build_geodesic_mesh
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent


def config() -> SimulationConfig:
    return SimulationConfig(subdivision=1, radial_cells=6, courant_factor=0.2)


def source() -> GaussianCurrent:
    return GaussianCurrent(peak_current_a=1.0e6)


def test_numpy_backend_defaults_to_cpu_float64() -> None:
    simulation = GeodesicFDTD(config=config())
    assert simulation.backend.name == "numpy"
    assert simulation.backend.device == "cpu"
    assert simulation.backend.dtype_name == "float64"
    assert simulation.er.dtype == np.float64


def test_backend_constants_do_not_alias_mesh_arrays() -> None:
    mesh = build_geodesic_mesh(1)
    simulation = GeodesicFDTD(config=config(), mesh=mesh)

    assert not np.shares_memory(simulation.backend.edges, mesh.edges)
    assert not np.shares_memory(simulation.backend.face_edges, mesh.face_edges)
    assert not np.shares_memory(
        simulation.backend.face_edge_signs, mesh.face_edge_signs
    )
    assert not np.shares_memory(
        simulation._primal_edge_angles, mesh.primal_edge_angles
    )


def test_numpy_backend_rejects_accelerator_device() -> None:
    with pytest.raises(BackendUnavailableError, match="only supports"):
        GeodesicFDTD(config=config(), backend="numpy", device="mps")


def test_numpy_backend_rejects_compiled_step() -> None:
    with pytest.raises(BackendUnavailableError, match="compiled field steps"):
        GeodesicFDTD(config=config(), backend="numpy", compile_step=True)


@pytest.mark.parametrize("chunk_size", (0, True, 1.5))
def test_compile_chunk_size_must_be_a_positive_integer(chunk_size) -> None:
    with pytest.raises(ValueError, match="compile_chunk_size"):
        GeodesicFDTD(config=config(), compile_chunk_size=chunk_size)


def test_compiled_step_batches_full_chunks_and_preserves_remainder(
    monkeypatch,
) -> None:
    calls = []

    def fake_compile_step(backend, function):
        del backend

        def compiled(values):
            calls.append(np.shape(values))
            return function(values)

        return compiled

    monkeypatch.setattr(NumPyBackend, "compile_step", fake_compile_step)
    eager = GeodesicFDTD(config=config(), source=source())
    compiled = GeodesicFDTD(
        config=config(),
        source=source(),
        compile_step=True,
        compile_chunk_size=8,
    )

    eager.step(19)
    compiled.step(19)

    assert calls == [(8,), (8,), (), (), ()]
    assert compiled.diagnostics()["compile_chunk_size"] == 8
    for field in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            compiled.to_numpy(getattr(compiled, field)),
            getattr(eager, field),
            rtol=5.0e-13,
            atol=5.0e-29,
        )


def test_numpy_backend_rejects_torch_threads() -> None:
    with pytest.raises(BackendUnavailableError, match="PyTorch CPU"):
        GeodesicFDTD(config=config(), backend="numpy", torch_threads=1)


def test_torch_backend_rejects_fractional_thread_count() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="positive integer"):
        GeodesicFDTD(
            config=config(),
            backend="torch",
            device="cpu",
            torch_threads=1.5,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("trailing_shape", [(), (7,), (3, 4)])
def test_numpy_incidence_circulation_matches_scatter(
    trailing_shape: tuple[int, ...],
) -> None:
    mesh = build_geodesic_mesh(1)
    backend = NumPyBackend(mesh)
    values = np.random.default_rng(42).standard_normal(
        (mesh.n_edges,) + trailing_shape
    )

    expected = mesh.dual_cell_circulation(values)
    actual = backend.dual_cell_circulation(values)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.count_nonzero(backend.vertex_edge_signs, axis=1).min() == 5
    assert np.count_nonzero(backend.vertex_edge_signs, axis=1).max() == 6


def test_torch_cpu_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    numpy_simulation = GeodesicFDTD(
        config=config(), source=source(), backend="numpy", dtype="float64"
    )
    torch_simulation = GeodesicFDTD(
        config=config(),
        source=source(),
        backend="torch",
        device="cpu",
        dtype="float64",
    )

    numpy_simulation.step(40)
    torch_simulation.step(40)

    assert torch_simulation.er.device.type == "cpu"
    assert torch_simulation.er.dtype == torch.float64
    for field in ("er", "et", "hr", "ht"):
        expected = getattr(numpy_simulation, field)
        actual = torch_simulation.to_numpy(getattr(torch_simulation, field))
        np.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=1.0e-12)


@pytest.mark.parametrize("trailing_shape", [(), (7,), (3, 4)])
def test_torch_face_circulation_matches_mesh(
    trailing_shape: tuple[int, ...],
) -> None:
    torch = pytest.importorskip("torch")
    mesh = build_geodesic_mesh(1)
    values = np.random.default_rng(42).standard_normal(
        (mesh.n_edges,) + trailing_shape
    )
    backend = GeodesicFDTD(
        config=config(), backend="torch", device="cpu", dtype="float64"
    ).backend

    actual = backend.to_numpy(backend.face_circulation(torch.asarray(values)))
    expected = mesh.face_circulation(values)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_float32_face_circulation_has_consistent_backend_precision() -> None:
    pytest.importorskip("torch")
    mesh = build_geodesic_mesh(1)
    numpy_backend = NumPyBackend(mesh, dtype="float32")
    torch_backend = GeodesicFDTD(
        config=config(), backend="torch", device="cpu", dtype="float32"
    ).backend
    values = np.zeros(mesh.n_edges, dtype=np.float32)
    face_edges = mesh.face_edges[0]
    values[face_edges] = (
        np.asarray((1.0e8, 1.0, -1.0e8), dtype=np.float32)
        * mesh.face_edge_signs[0]
    )

    numpy_result = numpy_backend.face_circulation(values)
    torch_result = torch_backend.to_numpy(
        torch_backend.face_circulation(torch_backend.asarray(values))
    )

    assert numpy_result.dtype == np.float32
    assert torch_result.dtype == np.float32
    np.testing.assert_array_equal(numpy_result, torch_result)


@pytest.mark.parametrize("trailing_shape", [(), (7,), (3, 4)])
def test_torch_dual_cell_circulation_matches_mesh(
    trailing_shape: tuple[int, ...],
) -> None:
    pytest.importorskip("torch")
    mesh = build_geodesic_mesh(1)
    values = np.random.default_rng(42).standard_normal(
        (mesh.n_edges,) + trailing_shape
    )
    backend = GeodesicFDTD(
        config=config(), backend="torch", device="cpu", dtype="float64"
    ).backend

    actual = backend.to_numpy(
        backend.dual_cell_circulation(backend.asarray(values))
    )
    expected = mesh.dual_cell_circulation(values)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_cuda_dual_cell_circulation_is_bitwise_repeatable() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = build_geodesic_mesh(3)
    simulation = GeodesicFDTD(
        config=SimulationConfig(
            subdivision=3, radial_cells=6, courant_factor=0.2
        ),
        backend="torch",
        device="cuda",
        dtype="float64",
    )
    values = simulation.backend.asarray(
        np.random.default_rng(20260805).standard_normal((mesh.n_edges, 7))
    )

    first = simulation.backend.dual_cell_circulation(values)
    for _ in range(10):
        repeated = simulation.backend.dual_cell_circulation(values)
        assert torch.equal(first, repeated)


def test_cuda_alias_is_pinned_to_construction_device() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    previous = torch.cuda.current_device()
    try:
        torch.cuda.set_device(1)
        simulation = GeodesicFDTD(
            config=config(), backend="torch", device="cuda", dtype="float64"
        )
        assert simulation.backend.device == "cuda:1"
        assert simulation.er.device == torch.device("cuda:1")

        torch.cuda.set_device(0)
        traces = simulation.record_er_observations(
            np.asarray(((0,),), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray(((1.0,),)),
            1,
        )
        assert traces.shape == (2, 1)
        assert traces.device == torch.device("cuda:1")
    finally:
        torch.cuda.set_device(previous)


def test_torch_auto_float32_tracks_float64_reference() -> None:
    torch = pytest.importorskip("torch")
    reference = GeodesicFDTD(
        config=config(), source=source(), backend="numpy", dtype="float64"
    )
    optimized = GeodesicFDTD(
        config=config(), source=source(), backend="torch", device="cpu"
    )

    reference.step(80)
    optimized.step(80)

    assert optimized.er.dtype == torch.float32
    assert optimized.memory_bytes * 2 == reference.memory_bytes
    for fields in (("er", "et"), ("hr", "ht")):
        expected = np.concatenate(
            tuple(getattr(reference, field).ravel() for field in fields)
        )
        actual = np.concatenate(
            tuple(
                optimized.to_numpy(getattr(optimized, field)).ravel()
                for field in fields
            )
        )
        relative_l2_error = np.linalg.norm(actual - expected) / np.linalg.norm(
            expected
        )
        assert relative_l2_error < 2.0e-5
    radial_magnetic_noise = np.max(
        np.abs(optimized.to_numpy(optimized.hr) - reference.hr)
    )
    assert radial_magnetic_noise < 1.0e-6 * np.max(np.abs(reference.ht))


def test_torch_compiled_cpu_matches_eager_with_source() -> None:
    pytest.importorskip("torch")
    eager = GeodesicFDTD(
        config=config(),
        source=source(),
        backend="torch",
        device="cpu",
        dtype="float64",
    )
    compiled = GeodesicFDTD(
        config=config(),
        source=source(),
        backend="torch",
        device="cpu",
        dtype="float64",
        compile_step=True,
    )

    eager.step(12)
    compiled.step(12)

    assert compiled.compiled
    assert compiled.steps == eager.steps
    assert compiled.time_s == pytest.approx(eager.time_s)
    for field in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            compiled.to_numpy(getattr(compiled, field)),
            eager.to_numpy(getattr(eager, field)),
            rtol=1.0e-11,
            atol=1.0e-12,
        )


def test_torch_compiled_nonuniform_stencil_matches_numpy() -> None:
    pytest.importorskip("torch")
    altitudes = (-10_000.0, -6_000.0, -2_000.0, -1_000.0, 0.0, 4_000.0)
    radial_config = SimulationConfig(
        subdivision=0,
        radial_cells=len(altitudes) - 1,
        minimum_altitude_m=altitudes[0],
        maximum_altitude_m=altitudes[-1],
        radial_altitudes_m=altitudes,
        radial_grid_policy="allow-abrupt",
        courant_factor=0.2,
    )
    reference = GeodesicFDTD(radial_config, backend="numpy", dtype="float64")
    compiled = GeodesicFDTD(
        radial_config,
        backend="torch",
        device="cpu",
        dtype="float64",
        compile_step=True,
    )
    generator = np.random.default_rng(20260805)
    for field in ("er", "et", "hr", "ht"):
        values = 1.0e-6 * generator.standard_normal(
            getattr(reference, field).shape
        )
        getattr(reference, field)[:] = values
        getattr(compiled, field).copy_(compiled.backend.asarray(values))

    reference.step(5)
    compiled.step(5)

    for field in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            compiled.to_numpy(getattr(compiled, field)),
            getattr(reference, field),
            rtol=2.0e-13,
            atol=1.0e-18,
        )


def test_torch_cpu_thread_count_is_configurable() -> None:
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    try:
        simulation = GeodesicFDTD(
            config=config(),
            backend="torch",
            device="cpu",
            torch_threads=1,
        )
        assert simulation.backend.threads == 1
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous_threads)


def test_torch_cpu_thread_metadata_tracks_process_global_state() -> None:
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    alternate = 1 if previous_threads != 1 else 2
    try:
        first = GeodesicFDTD(
            config=config(), backend="torch", device="cpu", torch_threads=1
        )
        second = GeodesicFDTD(
            config=config(),
            backend="torch",
            device="cpu",
            torch_threads=alternate,
        )

        assert first.backend.threads == alternate
        assert second.backend.threads == alternate
    finally:
        torch.set_num_threads(previous_threads)


def test_torch_auto_selects_an_available_device() -> None:
    torch = pytest.importorskip("torch")
    expected = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    simulation = GeodesicFDTD(config=config(), backend="torch", device="auto")
    assert simulation.er.device.type == expected
    assert simulation.er.dtype == torch.float32


def test_torch_fields_cross_the_visualization_boundary_as_numpy() -> None:
    pytest.importorskip("torch")
    from ionosphere_fdtd.visualization import _surface_values

    simulation = GeodesicFDTD(
        config=config(), source=source(), backend="torch", device="cpu"
    )
    simulation.step(2)
    values, _, association = _surface_values(simulation, "er", 0.0)
    assert isinstance(values, np.ndarray)
    assert association == "point"
    assert np.isfinite(values).all()


def test_torch_mps_runs_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        with pytest.raises(BackendUnavailableError, match="MPS"):
            GeodesicFDTD(config=config(), backend="torch", device="mps")
        return

    simulation = GeodesicFDTD(
        config=config(), source=source(), backend="torch", device="mps"
    )
    simulation.step(5)
    assert simulation.er.device.type == "mps"
    assert simulation.er.dtype == torch.float32
    assert np.isfinite(simulation.to_numpy(simulation.er)).all()


def test_torch_gpu_alias_uses_cuda_or_reports_unavailable() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        with pytest.raises(BackendUnavailableError, match="CUDA"):
            GeodesicFDTD(config=config(), backend="torch", device="gpu")
        return

    simulation = GeodesicFDTD(config=config(), backend="torch", device="gpu")
    assert simulation.er.device.type == "cuda"


def test_torch_mps_rejects_float64_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    with pytest.raises(BackendUnavailableError, match="does not support float64"):
        GeodesicFDTD(
            config=config(), backend="torch", device="mps", dtype="float64"
        )

import numpy as np
import pytest
import torch

from ionosphere_fdtd import BackendUnavailableError
from ionosphere_fdtd.mesh import build_geodesic_mesh
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent


def config() -> SimulationConfig:
    return SimulationConfig(subdivision=1, radial_cells=6, courant_factor=0.2)


def source() -> GaussianCurrent:
    return GaussianCurrent(peak_current_a=1.0e6)


def test_runtime_defaults_to_cpu_float64() -> None:
    simulation = GeodesicFDTD(config=config())

    assert simulation.runtime == "torch"
    assert simulation.device == torch.device("cpu")
    assert simulation.dtype == torch.float64
    assert simulation.dtype_name == "float64"
    assert simulation.er.device.type == "cpu"
    assert simulation.er.dtype == torch.float64


@pytest.mark.parametrize("chunk_size", (0, True, 1.5))
def test_compile_chunk_size_must_be_a_positive_integer(chunk_size) -> None:
    with pytest.raises(ValueError, match="compile_chunk_size"):
        GeodesicFDTD(config=config(), compile_chunk_size=chunk_size)


def test_runtime_rejects_fractional_thread_count() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GeodesicFDTD(
            config=config(),
            device="cpu",
            torch_threads=1.5,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("trailing_shape", [(), (7,), (3, 4)])
def test_face_circulation_matches_mesh(
    trailing_shape: tuple[int, ...],
) -> None:
    mesh = build_geodesic_mesh(1)
    values = np.random.default_rng(42).standard_normal(
        (mesh.n_edges,) + trailing_shape
    )
    simulation = GeodesicFDTD(
        config=config(), mesh=mesh, device="cpu", dtype="float64"
    )

    actual = simulation.to_numpy(
        simulation._runtime.face_circulation(
            simulation._runtime.as_tensor(values)
        )
    )
    expected = mesh.face_circulation(values)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("trailing_shape", [(), (7,), (3, 4)])
def test_dual_cell_circulation_matches_mesh(
    trailing_shape: tuple[int, ...],
) -> None:
    mesh = build_geodesic_mesh(1)
    values = np.random.default_rng(42).standard_normal(
        (mesh.n_edges,) + trailing_shape
    )
    simulation = GeodesicFDTD(
        config=config(), mesh=mesh, device="cpu", dtype="float64"
    )

    actual = simulation.to_numpy(
        simulation._runtime.dual_cell_circulation(
            simulation._runtime.as_tensor(values)
        )
    )
    expected = mesh.dual_cell_circulation(values)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_float32_tracks_float64_reference() -> None:
    reference = GeodesicFDTD(
        config=config(), source=source(), device="cpu", dtype="float64"
    )
    optimized = GeodesicFDTD(
        config=config(), source=source(), device="cpu", dtype="float32"
    )

    reference.step(80)
    optimized.step(80)

    assert optimized.er.dtype == torch.float32
    assert optimized.memory_bytes * 2 == reference.memory_bytes
    for fields in (("er", "et"), ("hr", "ht")):
        expected = np.concatenate(
            tuple(reference.to_numpy(getattr(reference, field)).ravel() for field in fields)
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


def test_compiled_cpu_matches_eager_with_source() -> None:
    eager = GeodesicFDTD(
        config=config(), source=source(), device="cpu", dtype="float64"
    )
    compiled = GeodesicFDTD(
        config=config(),
        source=source(),
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


def test_compiled_nonuniform_stencil_matches_eager() -> None:
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
    eager = GeodesicFDTD(radial_config, device="cpu", dtype="float64")
    compiled = GeodesicFDTD(
        radial_config, device="cpu", dtype="float64", compile_step=True
    )
    generator = np.random.default_rng(20260805)
    for field in ("er", "et", "hr", "ht"):
        values = 1.0e-6 * generator.standard_normal(getattr(eager, field).shape)
        getattr(eager, field).copy_(eager._runtime.as_tensor(values))
        getattr(compiled, field).copy_(compiled._runtime.as_tensor(values))

    eager.step(5)
    compiled.step(5)

    for field in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            compiled.to_numpy(getattr(compiled, field)),
            eager.to_numpy(getattr(eager, field)),
            rtol=2.0e-13,
            atol=1.0e-18,
        )


def test_cpu_thread_count_is_configurable() -> None:
    previous_threads = torch.get_num_threads()
    try:
        simulation = GeodesicFDTD(
            config=config(), device="cpu", torch_threads=1
        )
        assert simulation.threads == 1
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous_threads)


def test_cpu_thread_metadata_tracks_process_global_state() -> None:
    previous_threads = torch.get_num_threads()
    alternate = 1 if previous_threads != 1 else 2
    try:
        first = GeodesicFDTD(
            config=config(), device="cpu", torch_threads=1
        )
        second = GeodesicFDTD(
            config=config(), device="cpu", torch_threads=alternate
        )

        assert first.threads == alternate
        assert second.threads == alternate
    finally:
        torch.set_num_threads(previous_threads)


def test_auto_selects_an_available_device() -> None:
    expected = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    simulation = GeodesicFDTD(
        config=config(), device="auto", dtype="float32"
    )
    assert simulation.er.device.type == expected
    assert simulation.er.dtype == torch.float32


def test_fields_cross_visualization_boundary_as_numpy() -> None:
    from ionosphere_fdtd.visualization import _surface_values

    simulation = GeodesicFDTD(
        config=config(), source=source(), device="cpu"
    )
    simulation.step(2)
    values, _, association = _surface_values(simulation, "er", 0.0)
    assert isinstance(values, np.ndarray)
    assert association == "point"
    assert np.isfinite(values).all()


def test_mps_runs_when_available() -> None:
    if not torch.backends.mps.is_available():
        with pytest.raises(BackendUnavailableError, match="MPS"):
            GeodesicFDTD(config=config(), device="mps", dtype="float32")
        return

    simulation = GeodesicFDTD(
        config=config(), source=source(), device="mps", dtype="float32"
    )
    simulation.step(5)
    assert simulation.er.device.type == "mps"
    assert np.isfinite(simulation.to_numpy(simulation.er)).all()


def test_gpu_alias_uses_cuda_or_reports_unavailable() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(BackendUnavailableError, match="CUDA"):
            GeodesicFDTD(config=config(), device="gpu")
        return

    simulation = GeodesicFDTD(config=config(), device="gpu")
    assert simulation.er.device.type == "cuda"


def test_mps_rejects_float64_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    with pytest.raises(BackendUnavailableError, match="does not support float64"):
        GeodesicFDTD(config=config(), device="mps", dtype="float64")


def test_cuda_dual_cell_circulation_is_bitwise_repeatable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = build_geodesic_mesh(3)
    simulation = GeodesicFDTD(
        config=SimulationConfig(
            subdivision=3, radial_cells=6, courant_factor=0.2
        ),
        device="cuda",
        dtype="float64",
    )
    values = simulation._runtime.as_tensor(
        np.random.default_rng(20260805).standard_normal((mesh.n_edges, 7))
    )

    first = simulation._runtime.dual_cell_circulation(values)
    for _ in range(10):
        repeated = simulation._runtime.dual_cell_circulation(values)
        assert torch.equal(first, repeated)


def test_cuda_alias_is_pinned_to_construction_device() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    previous = torch.cuda.current_device()
    try:
        torch.cuda.set_device(1)
        simulation = GeodesicFDTD(
            config=config(), device="cuda", dtype="float64"
        )
        assert simulation.device == torch.device("cuda:1")
        assert simulation.er.device == torch.device("cuda:1")

        torch.cuda.set_device(0)
        traces = simulation.record_er_observations(
            np.asarray(((0,),), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray(((1.0,),)),
            1,
        )
        assert traces.device == torch.device("cuda:1")
    finally:
        torch.cuda.set_device(previous)

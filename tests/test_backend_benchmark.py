import pytest

from benchmarks.backend_matrix import run_backend_matrix
from benchmarks.backend_scaling import benchmark_cases
from benchmarks.distributed_scaling import summarize_distributed_timings
from benchmarks.torch_allocations import profile_allocations


def test_backend_matrix_always_reports_numpy_cpu() -> None:
    payload = run_backend_matrix(
        subdivision=0,
        radial_cells=2,
        steps=1,
        warmup_steps=0,
        repeats=1,
    )
    numpy_result = payload["results"][0]
    assert numpy_result["backend"] == "numpy"
    assert numpy_result["device"] == "cpu"
    assert numpy_result["status"] == "ok"
    assert numpy_result["steps_per_second"] > 0.0
    assert payload["configuration"]["torch_compile_chunk_size"] == 8
    assert numpy_result["compile_chunk_size"] == 8
    assert numpy_result["initialization_seconds"] > 0.0
    assert numpy_result["compile_seconds"] is None
    assert numpy_result["persistent_memory_bytes"] >= numpy_result["field_memory_bytes"]
    assert numpy_result["peak_process_memory_bytes"] > 0


def test_scaling_cases_use_only_eager_mode_for_numpy() -> None:
    cases = benchmark_cases(
        (2,),
        (16,),
        ("float32", "float64"),
        ("numpy", "torch-cpu", "cuda", "mps"),
        ("eager", "compiled"),
    )

    assert len(cases) == 14
    assert all(
        case["mode"] == "eager"
        for case in cases
        if case["implementation"] == "numpy"
    )
    assert {
        (case["backend"], case["device"])
        for case in cases
    } == {
        ("numpy", "cpu"),
        ("torch", "cpu"),
        ("torch", "cuda"),
        ("torch", "mps"),
    }


def test_distributed_timing_uses_median_slow_rank_duration() -> None:
    assert summarize_distributed_timings([4.0, 2.0, 3.0], 12) == {
        "median_seconds": 3.0,
        "steps_per_second": 4.0,
    }
    with pytest.raises(ValueError, match="positive"):
        summarize_distributed_timings([], 12)


def test_torch_allocation_profiler_reports_generic_cpu_operators() -> None:
    pytest.importorskip("torch")

    payload = profile_allocations(
        subdivision=0,
        radial_cells=2,
        dtype="float32",
        device="cpu",
        steps=1,
        warmup_steps=0,
    )

    assert payload["configuration"]["physics"] == "vacuum"
    assert payload["field_memory_bytes"] > 0
    assert payload["persistent_backend_bytes"] >= payload["field_memory_bytes"]
    assert payload["allocation_operators"]

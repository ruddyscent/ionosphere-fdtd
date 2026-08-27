import pytest
import torch

from benchmarks.backend_matrix import run_backend_matrix
from benchmarks.backend_scaling import benchmark_cases
from benchmarks.distributed_scaling import summarize_distributed_timings
from benchmarks.torch_allocations import profile_allocations


def test_runtime_matrix_always_reports_torch_cpu() -> None:
    payload = run_backend_matrix(
        subdivision=0,
        radial_cells=2,
        steps=1,
        warmup_steps=0,
        repeats=1,
    )
    cpu_result = payload["results"][0]
    assert cpu_result["runtime"] == "torch"
    assert cpu_result["device"] == "cpu"
    assert cpu_result["status"] == "ok"
    assert cpu_result["steps_per_second"] > 0.0
    assert payload["configuration"]["torch_compile_chunk_size"] == 8
    assert cpu_result["compile_chunk_size"] == 8
    assert cpu_result["initialization_seconds"] > 0.0
    assert cpu_result["compile_seconds"] is None
    assert cpu_result["persistent_memory_bytes"] >= cpu_result["field_memory_bytes"]
    assert cpu_result["peak_process_memory_bytes"] > 0


def test_scaling_cases_cover_pytorch_devices_and_modes() -> None:
    cases = benchmark_cases(
        (2,),
        (16,),
        ("float32", "float64"),
        ("torch-cpu", "cuda", "mps"),
        ("eager", "compiled"),
    )

    assert len(cases) == 12
    assert {
        (case["implementation"], case["device"])
        for case in cases
    } == {
        ("torch-cpu", "cpu"),
        ("cuda", "cuda"),
        ("mps", "mps"),
    }
    assert {case["mode"] for case in cases} == {"eager", "compiled"}


def test_distributed_timing_uses_median_slow_rank_duration() -> None:
    assert summarize_distributed_timings([4.0, 2.0, 3.0], 12) == {
        "median_seconds": 3.0,
        "steps_per_second": 4.0,
    }
    with pytest.raises(ValueError, match="positive"):
        summarize_distributed_timings([], 12)


def test_torch_allocation_profiler_reports_generic_cpu_operators() -> None:

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
    assert payload["persistent_runtime_bytes"] >= payload["field_memory_bytes"]
    assert payload["allocation_operators"]

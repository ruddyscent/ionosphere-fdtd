import pytest
import torch

from benchmarks.runtime_matrix import WORKLOADS, run_runtime_matrix
from benchmarks.runtime_scaling import (
    benchmark_cases,
    compare_pre_migration_baseline,
)
from benchmarks.distributed_scaling import summarize_distributed_timings
from benchmarks.torch_allocations import profile_allocations


def test_runtime_matrix_always_reports_torch_cpu() -> None:
    payload = run_runtime_matrix(
        subdivision=0,
        radial_cells=2,
        steps=1,
        warmup_steps=0,
        repeats=1,
    )
    cpu_result = payload["results"][0]
    assert payload["schema"] == "ionosphere-fdtd.pytorch-runtime-matrix.v1"
    assert cpu_result["runtime"] == "torch"
    assert cpu_result["device"] == "cpu"
    assert cpu_result["status"] == "ok"
    assert cpu_result["steps_per_second"] > 0.0
    assert payload["configuration"]["torch_compile_chunk_size"] == 8
    assert cpu_result["compile_chunk_size"] == 8
    assert cpu_result["initialization_seconds"] > 0.0
    assert cpu_result["cold_compile_seconds"] is None
    assert cpu_result["remainder_compile_seconds"] is None
    assert cpu_result["repeat_seconds"]
    assert cpu_result["workload"] == "bare"
    assert cpu_result["timing_scope"] == "bare-loop"
    assert cpu_result["persistent_memory_bytes"] >= cpu_result["field_memory_bytes"]
    assert cpu_result["peak_process_memory_bytes"] > 0


def test_scaling_cases_cover_pytorch_devices_and_modes() -> None:
    cases = benchmark_cases(
        ((2, 16),),
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
    assert {case["workload"] for case in cases} == {"bare"}


def test_runtime_matrix_measures_every_end_to_end_workload() -> None:
    payload = run_runtime_matrix(
        subdivision=0,
        radial_cells=2,
        steps=1,
        warmup_steps=0,
        repeats=1,
        workloads=WORKLOADS[1:],
    )

    cpu_results = [
        result for result in payload["results"] if result["device"] == "cpu"
    ]
    assert {result["workload"] for result in cpu_results} == set(WORKLOADS[1:])
    assert all(result["timing_scope"] == "end-to-end" for result in cpu_results)
    assert all(result["steps_per_second"] > 0.0 for result in cpu_results)


def test_pre_migration_comparison_matches_same_hardware_and_mode() -> None:
    current = [
        {
            "status": "ok",
            "workload": "bare",
            "device_name": "GPU",
            "subdivision": 2,
            "radial_cells": 16,
            "dtype": "float32",
            "mode": "eager",
            "steps_per_second": 96.0,
            "persistent_memory_bytes": 102,
        }
    ]
    baseline = {
        "measurements": [
            {
                "status": "ok",
                "backend": "torch",
                "machine": "GPU",
                "subdivision": 2,
                "radial_cells": 16,
                "dtype": "float32",
                "mode": "eager",
                "steps_per_second": 100.0,
                "persistent_memory_bytes": 100,
            }
        ]
    }

    comparison = compare_pre_migration_baseline(current, baseline)

    assert comparison[0]["within_tolerance"] is True
    assert comparison[0]["throughput_ratio"] == pytest.approx(0.96)


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

"""Measure PyTorch correctness-adjacent workloads across compute modes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path
import resource
import tempfile
from time import perf_counter

import numpy as np
import torch

from ionosphere_fdtd import BackendUnavailableError
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent


WORKLOADS = (
    "bare",
    "source-observation",
    "diagnostics-checkpoint",
    "surface-impedance",
    "plasma",
)


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    runtime: str
    device: str
    device_name: str | None
    dtype: str
    compiled: bool
    compile_chunk_size: int
    workload: str
    timing_scope: str
    status: str
    initialization_seconds: float | None
    cold_compile_seconds: float | None
    remainder_compile_seconds: float | None
    median_seconds: float | None
    repeat_seconds: tuple[float, ...] | None
    steps_per_second: float | None
    field_memory_bytes: int | None
    persistent_memory_bytes: int | None
    peak_process_memory_bytes: int | None
    peak_device_memory_bytes: int | None
    reason: str | None = None


class VacuumMaterial:
    def sample(self, directions, altitudes_m, earth_radius_m):
        del earth_radius_m
        shape = (len(directions), len(altitudes_m))
        return np.zeros(shape), np.ones(shape)


def run_runtime_matrix(
    *,
    subdivision: int = 2,
    radial_cells: int = 16,
    steps: int = 200,
    warmup_steps: int = 20,
    repeats: int = 3,
    dtype: str = "float32",
    torch_compile: bool = False,
    torch_compile_chunk_size: int = 8,
    workloads: tuple[str, ...] = ("bare",),
) -> dict[str, object]:
    """Run a PyTorch device/dtype/mode/workload matrix."""

    if min(steps, repeats, torch_compile_chunk_size) < 1 or warmup_steps < 0:
        raise ValueError("steps/repeats must be positive and warmup nonnegative")
    if not workloads or any(workload not in WORKLOADS for workload in workloads):
        raise ValueError(f"workloads must be drawn from {WORKLOADS}")
    results = [
        _measure(
            device,
            subdivision=subdivision,
            radial_cells=radial_cells,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            dtype=dtype,
            compile_step=torch_compile,
            compile_chunk_size=torch_compile_chunk_size,
            workload=workload,
        )
        for device in ("cpu", "cuda", "mps")
        for workload in workloads
    ]
    return {
        "schema": "ionosphere-fdtd.pytorch-runtime-matrix.v1",
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "configuration": {
            "subdivision": subdivision,
            "radial_cells": radial_cells,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "repeats": repeats,
            "dtype": dtype,
            "torch_compile": torch_compile,
            "torch_compile_chunk_size": torch_compile_chunk_size,
            "workloads": workloads,
        },
        "results": [asdict(result) for result in results],
    }


def _measure(
    device,
    *,
    subdivision,
    radial_cells,
    steps,
    warmup_steps,
    repeats,
    dtype,
    compile_step,
    compile_chunk_size,
    workload="bare",
):
    if workload not in WORKLOADS:
        raise ValueError(f"workload must be drawn from {WORKLOADS}")
    _reset_device_peak_memory(device)
    initialized = perf_counter()
    try:
        simulation = _create_simulation(
            subdivision=subdivision,
            radial_cells=radial_cells,
            dtype=dtype,
            device=device,
            compile_step=compile_step,
            compile_chunk_size=compile_chunk_size,
            workload=workload,
        )
        _initialize_fields(simulation)
        simulation._runtime.synchronize()
    except (BackendUnavailableError, RuntimeError) as error:
        return RuntimeResult(
            runtime="torch",
            device=device,
            device_name=None,
            dtype=dtype,
            compiled=compile_step,
            compile_chunk_size=compile_chunk_size,
            workload=workload,
            timing_scope=_timing_scope(workload),
            status="unavailable",
            initialization_seconds=None,
            cold_compile_seconds=None,
            remainder_compile_seconds=None,
            median_seconds=None,
            repeat_seconds=None,
            steps_per_second=None,
            field_memory_bytes=None,
            persistent_memory_bytes=None,
            peak_process_memory_bytes=_peak_process_memory_bytes(),
            peak_device_memory_bytes=_peak_device_memory_bytes(device),
            reason=str(error),
        )
    initialization_seconds = perf_counter() - initialized
    cold_compile_seconds = None
    remainder_compile_seconds = None
    if compile_step:
        compile_started = perf_counter()
        simulation.step(compile_chunk_size)
        simulation._runtime.synchronize()
        cold_compile_seconds = perf_counter() - compile_started
        remainder_started = perf_counter()
        simulation.step(1)
        simulation._runtime.synchronize()
        remainder_compile_seconds = perf_counter() - remainder_started
    temporary = tempfile.TemporaryDirectory(prefix="ionosphere-runtime-benchmark-")
    checkpoint_path = Path(temporary.name) / "checkpoint.npz"
    if warmup_steps:
        _run_workload(simulation, workload, warmup_steps, checkpoint_path)
        simulation._runtime.synchronize()
    elapsed = []
    for _ in range(repeats):
        started = perf_counter()
        _run_workload(simulation, workload, steps, checkpoint_path)
        simulation._runtime.synchronize()
        elapsed.append(perf_counter() - started)
    temporary.cleanup()
    median = float(np.median(elapsed))
    device_name = str(simulation.device)
    return RuntimeResult(
        runtime=simulation.runtime,
        device=device_name,
        device_name=_device_name(simulation),
        dtype=dtype,
        compiled=compile_step,
        compile_chunk_size=compile_chunk_size,
        workload=workload,
        timing_scope=_timing_scope(workload),
        status="ok",
        initialization_seconds=initialization_seconds,
        cold_compile_seconds=cold_compile_seconds,
        remainder_compile_seconds=remainder_compile_seconds,
        median_seconds=median,
        repeat_seconds=tuple(elapsed),
        steps_per_second=steps / median,
        field_memory_bytes=simulation.memory_bytes,
        persistent_memory_bytes=simulation.persistent_runtime_bytes,
        peak_process_memory_bytes=_peak_process_memory_bytes(),
        peak_device_memory_bytes=_peak_device_memory_bytes(device_name),
    )


def _create_simulation(
    *,
    subdivision: int,
    radial_cells: int,
    dtype: str,
    device: str,
    compile_step: bool,
    compile_chunk_size: int,
    workload: str,
) -> GeodesicFDTD:
    if workload in {"surface-impedance", "plasma"}:
        from benchmarks.torch_allocations import _simulation

        simulation = _simulation(
            subdivision=subdivision,
            radial_cells=radial_cells,
            dtype=dtype,
            device=device,
            physics=workload,
            compile_step=compile_step,
            compile_chunk_size=compile_chunk_size,
        )
        return simulation
    source = (
        GaussianCurrent(peak_current_a=1.0e6)
        if workload == "source-observation"
        else None
    )
    return GeodesicFDTD(
        SimulationConfig(
            subdivision=subdivision,
            radial_cells=radial_cells,
            minimum_altitude_m=0.0,
            maximum_altitude_m=100_000.0,
            courant_factor=0.35,
        ),
        material=(
            None if workload == "diagnostics-checkpoint" else VacuumMaterial()
        ),
        source=source,
        device=device,
        dtype=dtype,
        compile_step=compile_step,
        compile_chunk_size=compile_chunk_size,
    )


def _run_workload(
    simulation: GeodesicFDTD,
    workload: str,
    steps: int,
    checkpoint_path: Path,
) -> None:
    if workload == "source-observation":
        simulation.record_er_observations(
            np.asarray(((0,),), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray(((1.0,),)),
            steps,
        )
    else:
        simulation.step(steps)
    if workload == "diagnostics-checkpoint":
        simulation.diagnostics()
        simulation.save_checkpoint(checkpoint_path)


def _timing_scope(workload: str) -> str:
    return "bare-loop" if workload == "bare" else "end-to-end"


def _device_name(simulation: GeodesicFDTD) -> str:
    if simulation.device.type == "cuda":
        return simulation._runtime.torch.cuda.get_device_name(simulation.device)
    if simulation.device.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def _initialize_fields(simulation):
    generator = np.random.default_rng(20260814)
    for field in ("er", "et", "hr", "ht"):
        values = generator.standard_normal(getattr(simulation, field).shape) * 1.0e-6
        getattr(simulation, field)[:] = simulation._runtime.as_tensor(values)


def _peak_process_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _reset_device_peak_memory(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except (RuntimeError, ValueError):
        pass


def _peak_device_memory_bytes(device: str) -> int | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except (RuntimeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivision", type=int, default=2)
    parser.add_argument("--radial-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-chunk-size", type=int, default=8)
    parser.add_argument(
        "--workloads",
        default="bare",
        help=f"comma-separated workload names drawn from {WORKLOADS}",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    workloads = tuple(
        workload.strip() for workload in args.workloads.split(",") if workload.strip()
    )
    payload = run_runtime_matrix(
        subdivision=args.subdivision,
        radial_cells=args.radial_cells,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        dtype=args.dtype,
        torch_compile=args.torch_compile,
        torch_compile_chunk_size=args.torch_compile_chunk_size,
        workloads=workloads,
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure PyTorch device performance across production mesh and radial sizes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from benchmarks.runtime_matrix import WORKLOADS, _measure


IMPLEMENTATIONS = ("torch-cpu", "cuda", "mps")
MODES = ("eager", "compiled")


def benchmark_cases(
    grids: tuple[tuple[int, int], ...],
    dtypes: tuple[str, ...],
    implementations: tuple[str, ...],
    modes: tuple[str, ...],
    workloads: tuple[str, ...] = ("bare",),
) -> list[dict[str, Any]]:
    """Return deterministic, de-duplicated worker configurations."""

    cases = []
    for subdivision, radial in grids:
        for dtype in dtypes:
            for implementation in implementations:
                device = _device(implementation)
                for mode in modes:
                    for workload in workloads:
                        cases.append(
                            {
                                "subdivision": subdivision,
                                "radial_cells": radial,
                                "dtype": dtype,
                                "implementation": implementation,
                                "device": device,
                                "mode": mode,
                                "workload": workload,
                            }
                        )
    return cases


def _device(implementation: str) -> str:
    if implementation == "torch-cpu":
        return "cpu"
    if implementation in {"cuda", "mps"}:
        return implementation
    raise ValueError(f"unknown implementation: {implementation}")


def _case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["subdivision"],
        case["radial_cells"],
        case["dtype"],
        case["implementation"],
        case["mode"],
        case.get("workload", "bare"),
    )


def _worker_result(args: argparse.Namespace) -> dict[str, Any]:
    result = _measure(
        args.device,
        subdivision=args.subdivision,
        radial_cells=args.radial_cells,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        dtype=args.dtype,
        compile_step=args.mode == "compiled",
        compile_chunk_size=args.torch_compile_chunk_size,
        workload=args.workload,
    )
    payload = asdict(result)
    payload.update(
        subdivision=args.subdivision,
        radial_cells=args.radial_cells,
        implementation=args.implementation,
        mode=args.mode,
    )
    return payload


def _failed_result(
    case: dict[str, Any], status: str, reason: str, chunk_size: int
) -> dict[str, Any]:
    return {
        **case,
        "compiled": case["mode"] == "compiled",
        "compile_chunk_size": chunk_size,
        "device_name": None,
        "status": status,
        "initialization_seconds": None,
        "cold_compile_seconds": None,
        "remainder_compile_seconds": None,
        "median_seconds": None,
        "repeat_seconds": None,
        "steps_per_second": None,
        "field_memory_bytes": None,
        "persistent_memory_bytes": None,
        "peak_process_memory_bytes": None,
        "peak_device_memory_bytes": None,
        "reason": reason,
    }


def compare_pre_migration_baseline(
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare like-for-like bare Torch rows with historical measurements."""

    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must be in [0, 1)")
    comparisons = []
    for current in results:
        if current.get("status") != "ok" or current.get("workload") != "bare":
            continue
        match = next(
            (
                candidate
                for candidate in baseline.get("measurements", ())
                if candidate.get("status") == "ok"
                and _normalized_device_name(candidate.get("machine"))
                == _normalized_device_name(current.get("device_name"))
                and candidate.get("subdivision") == current.get("subdivision")
                and candidate.get("radial_cells") == current.get("radial_cells")
                and candidate.get("dtype") == current.get("dtype")
                and candidate.get("mode") == current.get("mode")
                and candidate.get("backend") == "torch"
            ),
            None,
        )
        if match is None:
            continue
        throughput_ratio = (
            current["steps_per_second"] / match["steps_per_second"]
        )
        persistent_ratio = (
            current["persistent_memory_bytes"] / match["persistent_memory_bytes"]
        )
        comparisons.append(
            {
                "subdivision": current["subdivision"],
                "radial_cells": current["radial_cells"],
                "dtype": current["dtype"],
                "mode": current["mode"],
                "device_name": current["device_name"],
                "baseline_steps_per_second": match["steps_per_second"],
                "current_steps_per_second": current["steps_per_second"],
                "throughput_ratio": throughput_ratio,
                "baseline_persistent_memory_bytes": match[
                    "persistent_memory_bytes"
                ],
                "current_persistent_memory_bytes": current[
                    "persistent_memory_bytes"
                ],
                "persistent_memory_ratio": persistent_ratio,
                "within_tolerance": (
                    throughput_ratio >= 1.0 - tolerance
                    and abs(persistent_ratio - 1.0) <= tolerance
                ),
            }
        )
    return comparisons


def _normalized_device_name(value: Any) -> str:
    name = str(value or "")
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _run_case(
    case: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.runtime_scaling",
        "--worker",
        "--subdivision",
        str(case["subdivision"]),
        "--radial-cells",
        str(case["radial_cells"]),
        "--dtype",
        case["dtype"],
        "--implementation",
        case["implementation"],
        "--device",
        case["device"],
        "--mode",
        case["mode"],
        "--workload",
        case["workload"],
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--repeats",
        str(args.repeats),
        "--torch-compile-chunk-size",
        str(args.torch_compile_chunk_size),
    ]
    environment = os.environ.copy()
    cache = None
    if args.cold_compile and case["mode"] == "compiled":
        cache = tempfile.TemporaryDirectory(prefix="ionosphere-inductor-")
        environment["TORCHINDUCTOR_CACHE_DIR"] = cache.name
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failed_result(
            case,
            "timeout",
            f"worker exceeded {args.timeout_seconds:g} seconds",
            args.torch_compile_chunk_size,
        )
    finally:
        if cache is not None:
            cache.cleanup()
    if completed.returncode != 0:
        reason = completed.stderr.strip() or f"worker exited {completed.returncode}"
        return _failed_result(
            case, "failed", reason[-2000:], args.torch_compile_chunk_size
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return _failed_result(
            case,
            "failed",
            f"invalid worker JSON: {error}: {completed.stdout[-1000:]}",
            args.torch_compile_chunk_size,
        )


def _system_information() -> dict[str, Any]:
    information: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, torch; print(json.dumps({"
            "'torch': torch.__version__, 'cuda': torch.version.cuda, "
            "'cuda_devices': [torch.cuda.get_device_name(i) "
            "for i in range(torch.cuda.device_count())], "
            "'mps_available': torch.backends.mps.is_available()}))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        information.update(json.loads(probe.stdout))
    else:
        information["torch"] = None
    return information


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _parse_csv(raw: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def _parse_grids(raw: str) -> tuple[tuple[int, int], ...]:
    try:
        grids = tuple(
            tuple(int(value) for value in item.strip().split(":"))
            for item in raw.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "grids must use subdivision:radial_cells pairs"
        ) from error
    if not grids or any(len(grid) != 2 or min(grid) < 1 for grid in grids):
        raise argparse.ArgumentTypeError(
            "grids must be positive subdivision:radial_cells pairs"
        )
    return grids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--grids",
        type=_parse_grids,
        default=_parse_grids("2:16,4:40,6:80,7:80"),
    )
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--implementations", default=",".join(IMPLEMENTATIONS))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--workloads", default="bare")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--torch-compile-chunk-size", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--cold-compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="historical torch-fast-path JSON used for like-for-like comparison",
    )
    parser.add_argument("--baseline-tolerance", type=float, default=0.05)
    parser.add_argument("--subdivision", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--radial-cells", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    parser.add_argument("--workload", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats, args.torch_compile_chunk_size) < 1:
        parser.error("steps, repeats, and compile chunk size must be positive")
    if args.warmup_steps < 0 or args.timeout_seconds <= 0.0:
        parser.error("warm-up must be nonnegative and timeout must be positive")
    if args.worker:
        print(json.dumps(_worker_result(args)))
        return 0
    if args.output is None:
        parser.error("--output is required for a scaling sweep")

    dtypes = _parse_csv(args.dtypes, str)
    implementations = _parse_csv(args.implementations, str)
    modes = _parse_csv(args.modes, str)
    workloads = _parse_csv(args.workloads, str)
    if any(value not in IMPLEMENTATIONS for value in implementations):
        parser.error(f"implementations must be drawn from {IMPLEMENTATIONS}")
    if any(value not in MODES for value in modes):
        parser.error(f"modes must be drawn from {MODES}")
    if any(value not in WORKLOADS for value in workloads):
        parser.error(f"workloads must be drawn from {WORKLOADS}")
    if any(value not in {"float32", "float64"} for value in dtypes):
        parser.error("dtypes must be float32 and/or float64")

    cases = benchmark_cases(
        args.grids, dtypes, implementations, modes, workloads
    )
    payload: dict[str, Any] = {
        "schema": "ionosphere-fdtd.pytorch-runtime-matrix.v1",
        "system": _system_information(),
        "configuration": {
            "grids": args.grids,
            "dtypes": dtypes,
            "implementations": implementations,
            "modes": modes,
            "workloads": workloads,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
            "torch_compile_chunk_size": args.torch_compile_chunk_size,
            "timeout_seconds": args.timeout_seconds,
            "cold_compile": args.cold_compile,
            "baseline_tolerance": args.baseline_tolerance,
        },
        "results": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        payload["results"] = previous.get("results", [])
    completed_keys = {_case_key(result) for result in payload["results"]}
    for position, case in enumerate(cases, start=1):
        if _case_key(case) in completed_keys:
            continue
        print(
            f"[{position}/{len(cases)}] "
            f"s{case['subdivision']} r{case['radial_cells']} "
            f"{case['dtype']} {case['implementation']} {case['mode']} "
            f"{case['workload']}",
            file=sys.stderr,
            flush=True,
        )
        payload["results"].append(_run_case(case, args))
        _write_payload(args.output, payload)
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        payload["baseline"] = {
            "path": str(args.baseline),
            "schema": baseline.get("schema"),
            "historical_only": True,
            "comparisons": compare_pre_migration_baseline(
                payload["results"],
                baseline,
                tolerance=args.baseline_tolerance,
            ),
        }
        _write_payload(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

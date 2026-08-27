"""Run one paired adaptive Simpson 2006 radar level on a single device."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np

from .model import (
    PAPER_FIGURE_7_DURATION_S,
    PAPER_SOURCE_CENTER_S,
    build_paper_adaptive_mesh,
    create_radar_simulation,
    record_radar_traces,
    save_radar_traces,
)


def _save_pair(traces_by_case: dict, output_dir: Path, target: int) -> None:
    if traces_by_case["reference"].run_signature != traces_by_case["anomaly"].run_signature:
        raise ValueError("adaptive reference/anomaly signatures do not match")
    output_dir.mkdir(parents=True, exist_ok=True)
    for case, traces in traces_by_case.items():
        output = save_radar_traces(traces, output_dir / f"s{target}-{case}.npz")
        print(f"output={output}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-subdivision", type=int, default=7)
    parser.add_argument("--target-subdivision", type=int, required=True)
    parser.add_argument("--core-radius-deg", type=float, default=1.0)
    parser.add_argument("--transition-width-deg", type=float, default=1.0)
    parser.add_argument(
        "--material", choices=("etopo5", "natural-earth"), default="etopo5"
    )
    parser.add_argument("--etopo5-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--compile-chunk-size", type=int, default=32)
    parser.add_argument("--sample-every", type=int, default=32)
    parser.add_argument(
        "--stop-after-center", type=float, default=PAPER_FIGURE_7_DURATION_S
    )
    parser.add_argument("--synchronize-every", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.material == "etopo5" and args.etopo5_path is None:
        raise SystemExit("--etopo5-path is required with --material etopo5")
    if args.target_subdivision <= args.base_subdivision:
        raise SystemExit("--target-subdivision must exceed --base-subdivision")
    if args.compile_chunk_size < 1 or args.sample_every < 1:
        raise SystemExit("compile and sample chunk sizes must be positive")
    if args.stop_after_center <= 0.0:
        raise SystemExit("--stop-after-center must be positive")

    mesh = build_paper_adaptive_mesh(
        args.target_subdivision,
        base_subdivision=args.base_subdivision,
        core_radius_deg=args.core_radius_deg,
        transition_width_deg=args.transition_width_deg,
    )
    traces_by_case = {}
    for case in ("reference", "anomaly"):
        simulation = create_radar_simulation(
            include_oil=case == "anomaly",
            subdivision=args.base_subdivision,
            material_model=args.material,
            etopo5_path=args.etopo5_path,
            device=args.device,
            dtype=args.dtype,
            compile_step=True,
            compile_chunk_size=args.compile_chunk_size,
            mesh=mesh,
            lithosphere_profile="figure-15",
            ionosphere_model="day-night",
        )
        steps = int(
            np.ceil(
                (PAPER_SOURCE_CENTER_S + args.stop_after_center)
                / simulation.time_step_s
            )
        )
        print(
            f"case={case} target=s{args.target_subdivision} "
            f"faces={mesh.n_faces:,} dt={simulation.time_step_s:.9e}s "
            f"steps={steps:,} device={simulation.device}",
            flush=True,
        )
        started = time.perf_counter()
        traces = record_radar_traces(
            simulation,
            steps=steps,
            case=case,
            synchronize_every=args.synchronize_every,
            sample_every=args.sample_every,
        )
        print(
            f"elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
        traces_by_case[case] = traces
        del simulation
        gc.collect()
    _save_pair(traces_by_case, args.output_dir, args.target_subdivision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

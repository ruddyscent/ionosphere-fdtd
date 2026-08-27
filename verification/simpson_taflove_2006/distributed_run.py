"""Run a paired adaptive Simpson 2006 radar case with two torchrun ranks."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from ionosphere_fdtd.distributed import (
    DistributedGeodesicFDTD,
    initialize_torchrun_process_group,
)
from ionosphere_fdtd.partition import partition_surface_mesh

from .model import (
    PAPER_FIGURE_7_DURATION_S,
    PAPER_OIL_LATITUDE_DEG,
    PAPER_OIL_LONGITUDE_DEG,
    PAPER_SOURCE_CENTER_S,
    PAPER_TRANSMITTER_LATITUDE_DEG,
    PAPER_TRANSMITTER_LONGITUDE_DEG,
    build_paper_adaptive_mesh,
    create_radar_simulation,
    record_radar_traces,
    save_radar_traces,
)


def _direction(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    return np.asarray(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-subdivision", type=int, default=7)
    parser.add_argument("--target-subdivision", type=int, default=10)
    parser.add_argument("--core-radius-deg", type=float, default=1.0)
    parser.add_argument("--transition-width-deg", type=float, default=1.0)
    parser.add_argument(
        "--material", choices=("etopo5", "natural-earth"), default="etopo5"
    )
    parser.add_argument("--etopo5-path", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--capacities", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--cuda-graph-chunk-size", type=int, default=32)
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
    if not args.base_subdivision < args.target_subdivision:
        raise SystemExit("--target-subdivision must exceed --base-subdivision")
    if args.stop_after_center <= 0.0:
        raise SystemExit("--stop-after-center must be positive")
    if args.cuda_graph_chunk_size < 0:
        raise SystemExit("--cuda-graph-chunk-size must be nonnegative")
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be positive")

    device = initialize_torchrun_process_group("nccl")
    import torch.distributed as distributed

    rank = distributed.get_rank()
    try:
        mesh = build_paper_adaptive_mesh(
            args.target_subdivision,
            base_subdivision=args.base_subdivision,
            core_radius_deg=args.core_radius_deg,
            transition_width_deg=args.transition_width_deg,
        )
        seeds = np.stack(
            (
                _direction(
                    PAPER_TRANSMITTER_LATITUDE_DEG,
                    PAPER_TRANSMITTER_LONGITUDE_DEG,
                ),
                _direction(PAPER_OIL_LATITUDE_DEG, PAPER_OIL_LONGITUDE_DEG),
            )
        )
        partition = partition_surface_mesh(
            mesh,
            seed_directions=seeds,
            part_capacities=np.asarray(args.capacities),
        )
        traces = {}
        for case in ("reference", "anomaly"):
            setup = create_radar_simulation(
                include_oil=case == "anomaly",
                subdivision=args.base_subdivision,
                material_model=args.material,
                etopo5_path=args.etopo5_path,
                dtype="float64",
                compile_step=False,
                mesh=mesh,
                lithosphere_profile="figure-15",
                ionosphere_model="day-night",
            )
            simulation = DistributedGeodesicFDTD(
                partition,
                config=setup.config,
                mesh=mesh,
                material=setup.material,
                source=setup.source,
                device=str(device),
                dtype=args.dtype,
            )
            try:
                simulation.radar_receiver_altitude_m = (
                    setup.radar_receiver_altitude_m
                )
                simulation.radar_vertical_reference = (
                    setup.radar_vertical_reference
                )
                if args.cuda_graph_chunk_size:
                    simulation.enable_cuda_graph(args.cuda_graph_chunk_size)
                steps = int(
                    np.ceil(
                        (PAPER_SOURCE_CENTER_S + args.stop_after_center)
                        / simulation.time_step_s
                    )
                )
                if rank == 0:
                    print(
                        f"case={case} target=s{args.target_subdivision} "
                        f"faces={mesh.n_faces:,} "
                        f"dt={simulation.time_step_s:.9e}s steps={steps:,}",
                        flush=True,
                    )
                traces[case] = record_radar_traces(
                    simulation,
                    steps=steps,
                    case=case,
                    synchronize_every=args.synchronize_every,
                    sample_every=args.sample_every,
                )
            finally:
                simulation.close()
            del simulation, setup
            gc.collect()
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for case, values in traces.items():
                output = save_radar_traces(
                    values,
                    args.output_dir
                    / f"s{args.target_subdivision}-{case}.npz",
                )
                print(f"output={output}", flush=True)
        distributed.barrier()
    finally:
        distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

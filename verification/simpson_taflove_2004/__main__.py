"""Run the Simpson–Taflove 2004 Figure 7/8 validation experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

from ionosphere_fdtd import BackendUnavailableError

from ..common.archive import save_npz_atomic
from ..mesh_optimization.mesquite import load_optimized_mesh
from ..physics_diagnostics import (
    TensorBoardPhysicsRecorder,
    save_physics_snapshots,
)

from .model import (
    PAPER_DFT_TRUNCATIONS,
    PAPER_MINIMUM_SIMULATION_STEPS,
    PAPER_TRACE_STEPS,
    REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M,
    REPRESENTATIVE_IONOSPHERE_REFERENCE_HEIGHT_M,
    REPRESENTATIVE_IONOSPHERE_SCALE_HEIGHT_M,
    arrival_metrics,
    compute_attenuation,
    compute_phase_velocity,
    create_validation_simulation,
    equatorial_path_diagnostic_regions,
    phase_velocity_metrics,
    record_validation_traces,
    render_figure_7,
    render_figure_8,
    source_distribution_metrics,
    trace_metrics,
    validation_metrics,
)
from .report import (
    ValidationRunSummary,
    write_validation_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/simpson-taflove-2004"),
    )
    parser.add_argument("--subdivision", type=int, choices=range(0, 9), default=7)
    parser.add_argument(
        "--mesh-orientation", choices=("native", "polar"), default="polar"
    )
    parser.add_argument(
        "--mesh-optimization-steps",
        type=int,
        default=0,
        help="apply deterministic spherical edge-quality optimization",
    )
    parser.add_argument(
        "--mesh-coordinates",
        type=Path,
        help="NPZ coordinates produced by verification.mesh_optimization",
    )
    parser.add_argument("--steps", type=int, default=PAPER_TRACE_STEPS)
    parser.add_argument(
        "--material",
        choices=("natural-earth", "etopo5", "uniform"),
        default="natural-earth",
    )
    parser.add_argument(
        "--etopo5-path",
        type=Path,
        help="NOAA-NGDC big-endian ETOPO5.DAT (required by --material etopo5)",
    )
    parser.add_argument(
        "--radial-support",
        choices=("point", "dual-cell"),
        default="point",
        help="sample Er material at one vertex or average its dual-cell area",
    )
    parser.add_argument(
        "--tangential-interface",
        choices=("point", "fractional"),
        default="point",
        help="sample Et materials at cell centers or average radial interfaces",
    )
    parser.add_argument(
        "--tangential-support",
        choices=("point", "edge-diamond"),
        default="point",
        help="sample Et material at one edge point or four dual-support points",
    )
    parser.add_argument(
        "--minimum-ocean-depth-km",
        type=float,
        default=0.0,
        help="opt-in conservative ocean-column depth for radial voxelization",
    )
    parser.add_argument(
        "--deep-lithosphere-resistivity-ohm-m",
        type=float,
        default=REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M,
        help="representative resistivity below 60 km from Hermance Figure 6",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--synchronize-every", type=int, default=128)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        help="write periodic physics diagnostics as TensorBoard event files",
    )
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=512,
        help="TensorBoard physics sampling interval in time steps",
    )
    parser.add_argument(
        "--dft-window",
        choices=("adaptive", "paper"),
        default="adaptive",
        help="truncate at each simulated zero crossing or use the paper's samples",
    )
    parser.add_argument(
        "--spectral-window",
        choices=("rectangular", "cosine-tail"),
        default="rectangular",
        help="apply no taper or a 10%% terminal cosine taper before the DFT",
    )
    parser.add_argument(
        "--ionosphere-reference-height-km",
        type=float,
        default=REPRESENTATIVE_IONOSPHERE_REFERENCE_HEIGHT_M / 1_000.0,
    )
    parser.add_argument(
        "--ionosphere-scale-height-km",
        type=float,
        default=REPRESENTATIVE_IONOSPHERE_SCALE_HEIGHT_M / 1_000.0,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Markdown report path (default: OUTPUT_DIR/verification-report.md)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps < PAPER_MINIMUM_SIMULATION_STEPS:
        raise SystemExit(
            "--steps must be at least "
            f"{PAPER_MINIMUM_SIMULATION_STEPS} for the validation DFT windows"
        )
    started = time.perf_counter()
    if args.mesh_coordinates is not None and args.mesh_optimization_steps:
        raise SystemExit(
            "--mesh-coordinates cannot be combined with --mesh-optimization-steps"
        )
    optimized_mesh = None
    mesh_metadata = None
    if args.mesh_coordinates is not None:
        try:
            optimized_mesh, mesh_metadata = load_optimized_mesh(
                args.mesh_coordinates,
                expected_subdivision=args.subdivision,
                expected_orientation=args.mesh_orientation,
            )
        except (OSError, KeyError, ValueError) as error:
            raise SystemExit(str(error)) from error
    try:
        simulation = create_validation_simulation(
            subdivision=args.subdivision,
            material_model=args.material,
            device=args.device,
            dtype=args.dtype,
            compile_step=args.torch_compile,
            torch_threads=args.torch_threads,
            mesh_orientation=args.mesh_orientation,
            mesh_optimization_steps=args.mesh_optimization_steps,
            ionosphere_reference_height_m=(
                1_000.0 * args.ionosphere_reference_height_km
            ),
            ionosphere_scale_height_m=1_000.0 * args.ionosphere_scale_height_km,
            etopo5_path=args.etopo5_path,
            radial_material_support=args.radial_support,
            tangential_interface_mode=args.tangential_interface,
            tangential_material_support=args.tangential_support,
            minimum_ocean_depth_m=1_000.0 * args.minimum_ocean_depth_km,
            deep_lithosphere_resistivity_ohm_m=(
                args.deep_lithosphere_resistivity_ohm_m
            ),
            mesh=optimized_mesh,
        )
    except (BackendUnavailableError, ImportError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"grid={simulation.mesh.n_vertices:,}x{simulation.config.radial_cells} "
        f"runtime={simulation.runtime} device={simulation.device} "
        f"dtype={simulation.dtype_name} material={args.material} "
        f"mesh_optimization_steps={args.mesh_optimization_steps} "
        f"mesh_coordinates={args.mesh_coordinates or 'generated'} "
        f"minimum_ocean_depth_km={args.minimum_ocean_depth_km:g} "
        "deep_lithosphere_resistivity_ohm_m="
        f"{args.deep_lithosphere_resistivity_ohm_m:g} "
        f"radial_support={args.radial_support} "
        f"tangential_interface={args.tangential_interface} "
        f"tangential_support={args.tangential_support} "
        f"dt={simulation.time_step_s:.3e}s",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_recorder = None
    diagnostic_data = None
    if args.tensorboard_log_dir is not None:
        if args.diagnostics_every < 1:
            raise SystemExit("--diagnostics-every must be positive")
        try:
            diagnostic_recorder = TensorBoardPhysicsRecorder(
                simulation,
                args.tensorboard_log_dir,
                horizontal_regions=equatorial_path_diagnostic_regions(
                    simulation
                ),
                metadata={
                    "study": "Simpson-Taflove 2004 Figures 7-8",
                    "material": args.material,
                    "mesh_orientation": args.mesh_orientation,
                    "mesh_coordinates": str(
                        args.mesh_coordinates or "generated"
                    ),
                    "steps": args.steps,
                    "diagnostics_every": args.diagnostics_every,
                    "diagnostic_corridor_half_width_deg": 10.0,
                    "diagnostic_source_exclusion_deg": 5.0,
                    "radial_support": args.radial_support,
                    "tangential_interface": args.tangential_interface,
                    "tangential_support": args.tangential_support,
                    "minimum_ocean_depth_km": args.minimum_ocean_depth_km,
                    "deep_lithosphere_resistivity_ohm_m": (
                        args.deep_lithosphere_resistivity_ohm_m
                    ),
                    "ionosphere_reference_height_km": (
                        args.ionosphere_reference_height_km
                    ),
                    "ionosphere_scale_height_km": (
                        args.ionosphere_scale_height_km
                    ),
                },
            )
        except ImportError as error:
            raise SystemExit(str(error)) from error
        print(
            "physics_diagnostic_backend_bytes="
            f"{diagnostic_recorder.sampler.diagnostic_backend_bytes:,} "
            f"tensorboard={args.tensorboard_log_dir}",
            flush=True,
        )
    try:
        traces = record_validation_traces(
            simulation,
            steps=args.steps,
            synchronize_every=args.synchronize_every,
            diagnostics_every=args.diagnostics_every,
            recorder=diagnostic_recorder,
        )
    finally:
        if diagnostic_recorder is not None:
            diagnostic_recorder.close()
    if diagnostic_recorder is not None:
        diagnostic_data = save_physics_snapshots(
            args.output_dir / "physics-diagnostics.npz",
            diagnostic_recorder.snapshots,
            node_altitudes_m=simulation.altitudes_m,
            cell_altitudes_m=simulation.radial_midpoint_altitudes_m,
            metadata=diagnostic_recorder.metadata,
        )
    trace_data = save_npz_atomic(
        args.output_dir / "simpson-taflove-2004-traces.npz",
        time_steps=traces.time_steps,
        time_s=traces.time_s,
        er_v_m=traces.er_v_m,
        labels=np.asarray(traces.labels),
    )
    figure_7 = render_figure_7(
        traces,
        args.output_dir / "simpson-taflove-2004-fig-7.png",
    )
    try:
        curves = compute_attenuation(
            traces,
            truncations=(
                PAPER_DFT_TRUNCATIONS if args.dft_window == "paper" else None
            ),
            spectral_window=args.spectral_window,
        )
    except ValueError as error:
        raise SystemExit(f"invalid DFT window: {error}") from error
    figure_8 = render_figure_8(
        curves,
        args.output_dir / "simpson-taflove-2004-fig-8.png",
    )
    metrics = validation_metrics(curves)
    metrics.update(trace_metrics(traces))
    metrics.update(arrival_metrics(traces))
    metrics.update(source_distribution_metrics(simulation))
    metrics.update(
        phase_velocity_metrics(
            compute_phase_velocity(
                traces,
                truncations=(
                    PAPER_DFT_TRUNCATIONS if args.dft_window == "paper" else None
                ),
                spectral_window=args.spectral_window,
            )
        )
    )
    if mesh_metadata is not None:
        metrics.update(
            {
                "mesh_mesquite_maximum_displacement_rad": float(
                    mesh_metadata["maximum_displacement_rad"]
                ),
                "mesh_mesquite_laplace_l1_relative_l2": float(
                    mesh_metadata["quality_after"][
                        "laplace_l1_max_relative_l2"
                    ]
                ),
                "mesh_mesquite_laplace_l2_relative_l2": float(
                    mesh_metadata["quality_after"][
                        "laplace_l2_max_relative_l2"
                    ]
                ),
            }
        )
    metrics.update(
        {
            f"{label}_dft_cutoff_step": cutoff
            for label, cutoff in curves.dft_truncations.items()
        }
    )
    elapsed_s = time.perf_counter() - started
    report = write_validation_report(
        ValidationRunSummary(
            generated_at=datetime.now().astimezone(),
            command=_reproduction_command(args),
            git_revision=_git_revision(),
            subdivision=args.subdivision,
            mesh_optimization_steps=args.mesh_optimization_steps,
            minimum_ocean_depth_m=1_000.0 * args.minimum_ocean_depth_km,
            deep_lithosphere_resistivity_ohm_m=(
                args.deep_lithosphere_resistivity_ohm_m
            ),
            surface_cells=simulation.mesh.n_vertices,
            radial_cells=simulation.config.radial_cells,
            time_step_s=simulation.time_step_s,
            steps=args.steps,
            material_model=args.material,
            relief_data=args.etopo5_path,
            ionosphere_reference_height_m=(
                1_000.0 * args.ionosphere_reference_height_km
            ),
            ionosphere_scale_height_m=1_000.0 * args.ionosphere_scale_height_km,
            dft_window=args.dft_window,
            spectral_window=args.spectral_window,
            radial_support=args.radial_support,
            tangential_interface=args.tangential_interface,
            tangential_support=args.tangential_support,
            runtime=simulation.runtime,
            device=simulation.device,
            dtype=simulation.dtype_name,
            compiled=simulation.compiled,
            elapsed_s=elapsed_s,
            metrics=metrics,
            figure_7=figure_7,
            figure_8=figure_8,
            trace_data=trace_data,
        ),
        args.report or args.output_dir / "verification-report.md",
    )
    print(f"figure 7: {figure_7}")
    print(f"figure 8: {figure_8}")
    print(f"traces: {trace_data}")
    if diagnostic_data is not None:
        print(f"physics diagnostics: {diagnostic_data}")
        print(f"TensorBoard logs: {args.tensorboard_log_dir}")
    print(f"report: {report}")
    for name, value in metrics.items():
        rendered = str(value) if isinstance(value, int) else f"{value:.3f}"
        print(f"{name}: {rendered}")
    print(f"elapsed: {elapsed_s:.1f}s")
    return 0


def _reproduction_command(args: argparse.Namespace) -> str:
    compile_flag = "--torch-compile" if args.torch_compile else "--no-torch-compile"
    quote = shlex.quote
    tensorboard_dependency = (
        "--extra tensorboard " if args.tensorboard_log_dir is not None else ""
    )
    parts = [
        f"uv run {tensorboard_dependency}"
        "--extra visualization python -m "
        "verification.simpson_taflove_2004",
        f"--subdivision {args.subdivision}",
        f"--mesh-orientation {quote(args.mesh_orientation)}",
        f"--mesh-optimization-steps {args.mesh_optimization_steps}",
        f"--minimum-ocean-depth-km {args.minimum_ocean_depth_km:g}",
        "--deep-lithosphere-resistivity-ohm-m "
        f"{args.deep_lithosphere_resistivity_ohm_m:g}",
        f"--steps {args.steps}",
        f"--material {quote(args.material)}",
        f"--device {quote(args.device)}",
        f"--dtype {quote(args.dtype)}",
        f"--dft-window {quote(args.dft_window)}",
        f"--spectral-window {quote(args.spectral_window)}",
        f"--radial-support {quote(args.radial_support)}",
        f"--tangential-interface {quote(args.tangential_interface)}",
        f"--tangential-support {quote(args.tangential_support)}",
        "--ionosphere-reference-height-km "
        f"{args.ionosphere_reference_height_km:g}",
        f"--ionosphere-scale-height-km {args.ionosphere_scale_height_km:g}",
        compile_flag,
        f"--synchronize-every {args.synchronize_every}",
        f"--output-dir {quote(str(args.output_dir))}",
    ]
    if args.torch_threads is not None:
        parts.append(f"--torch-threads {args.torch_threads}")
    if args.etopo5_path is not None:
        parts.append(f"--etopo5-path {quote(str(args.etopo5_path))}")
    if args.mesh_coordinates is not None:
        parts.append(f"--mesh-coordinates {quote(str(args.mesh_coordinates))}")
    if args.report is not None:
        parts.append(f"--report {quote(str(args.report))}")
    if args.tensorboard_log_dir is not None:
        parts.extend(
            (
                f"--tensorboard-log-dir {quote(str(args.tensorboard_log_dir))}",
                f"--diagnostics-every {args.diagnostics_every}",
            )
        )
    separator = f" {chr(92)}\n  "
    return separator.join(parts)


def _git_revision() -> str:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "--short", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision}-dirty" if dirty else revision


if __name__ == "__main__":
    raise SystemExit(main())

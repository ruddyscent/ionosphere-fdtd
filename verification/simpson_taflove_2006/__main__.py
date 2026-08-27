"""Run the Simpson--Taflove 2006 Figure 5--7 verification experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import time
from pathlib import Path

import numpy as np

from ..mesh_optimization.mesquite import load_optimized_mesh
from ..simpson_taflove_2004.model import ValidationTraces, compute_attenuation
from .model import (
    PAPER_FIGURE_7_DURATION_S,
    PAPER_SOURCE_CENTER_S,
    REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M,
    THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
    build_paper_adaptive_mesh,
    compare_radar_resolution_pairs,
    compute_radar_perturbation,
    create_radar_simulation,
    load_radar_traces,
    radar_field_metrics,
    radar_metrics,
    record_radar_traces,
    render_figure_5,
    render_figure_6,
    render_figure_7,
    save_radar_traces,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fig56 = commands.add_parser("figures-5-6")
    fig56.add_argument("--traces", type=Path, required=True)
    fig56.add_argument("--output-dir", type=Path, required=True)

    radar = commands.add_parser("radar-run")
    radar.add_argument("--case", choices=("reference", "anomaly"), required=True)
    radar.add_argument("--output", type=Path, required=True)
    radar.add_argument("--subdivision", type=int, choices=range(8), default=7)
    radar.add_argument(
        "--mesh-orientation", choices=("native", "polar"), default="polar"
    )
    radar.add_argument(
        "--mesh-optimization-steps",
        type=int,
        default=0,
        help="apply deterministic spherical edge-quality optimization",
    )
    radar.add_argument(
        "--mesh-coordinates",
        type=Path,
        help="NPZ coordinates produced by verification.mesh_optimization",
    )
    radar.add_argument(
        "--geometry-mode",
        choices=("thin-shell", "full-spherical"),
        default="full-spherical",
    )
    radar.add_argument(
        "--material", choices=("etopo5", "natural-earth"), default="etopo5"
    )
    radar.add_argument("--etopo5-path", type=Path)
    radar.add_argument(
        "--deep-lithosphere-resistivity-ohm-m",
        type=float,
        default=REPRESENTATIVE_DEEP_LITHOSPHERE_RESISTIVITY_OHM_M,
    )
    radar.add_argument(
        "--upper-crust-resistivity-ohm-m", type=float, default=500.0
    )
    radar.add_argument(
        "--asthenosphere-resistivity-ohm-m", type=float, default=200.0
    )
    radar.add_argument(
        "--lithosphere-profile",
        choices=("legacy", "figure-15"),
        default="legacy",
    )
    radar.add_argument(
        "--ionosphere", choices=("daytime", "day-night"), default="daytime"
    )
    radar.add_argument("--subsolar-latitude-deg", type=float, default=0.0)
    radar.add_argument(
        "--subsolar-longitude-deg",
        type=float,
        default=THESIS_DAWN_ALIGNED_SUBSOLAR_LONGITUDE_DEG,
        help="90 degrees places the dawn terminator at 0 degrees longitude",
    )
    radar.add_argument(
        "--tangential-interface",
        choices=("point", "fractional"),
        default="point",
    )
    radar.add_argument(
        "--tangential-support",
        choices=("point", "edge-diamond"),
        default="point",
    )
    radar.add_argument("--device", default="auto")
    radar.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    radar.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=True
    )
    radar.add_argument("--torch-compile-chunk-size", type=int, default=8)
    radar.add_argument("--source-center", type=float, default=PAPER_SOURCE_CENTER_S)
    radar.add_argument(
        "--source-altitude-m",
        type=float,
        help="override source altitude; otherwise use --vertical-reference",
    )
    radar.add_argument(
        "--vertical-reference",
        choices=("terrain", "sea-level"),
        default="terrain",
    )
    radar.add_argument(
        "--source-basis",
        choices=("both", "north", "east", "difference"),
        default="both",
    )
    radar.add_argument("--courant", type=float, default=0.4)
    radar.add_argument(
        "--source-edge-assignment",
        choices=("projected", "nearest"),
        default="projected",
    )
    radar.add_argument(
        "--stop-after-center", type=float, default=PAPER_FIGURE_7_DURATION_S
    )
    radar.add_argument("--synchronize-every", type=int, default=256)
    radar.add_argument("--sample-every", type=int, default=1)
    radar.add_argument(
        "--receiver-support",
        choices=("face", "local-linear"),
        default="local-linear",
    )
    radar.add_argument(
        "--shield", action=argparse.BooleanOptionalAction, default=True
    )
    radar.add_argument("--shield-radius-km", type=float, default=2_500.0)
    radar.add_argument(
        "--horizontal-anomaly",
        choices=("conservative-nearest", "point"),
        default="conservative-nearest",
    )

    analyze = commands.add_parser("analyze-radar")
    analyze.add_argument("--reference", type=Path, required=True)
    analyze.add_argument("--anomaly", type=Path, required=True)
    analyze.add_argument("--figure", type=Path, required=True)
    analyze.add_argument(
        "--normalization", choices=("pointwise", "peak"), default="pointwise"
    )
    analyze.add_argument(
        "--ht-definition",
        choices=(
            "principal-axis",
            "east",
            "north",
            "magnitude",
            "vector-difference",
        ),
        default="vector-difference",
    )

    convergence = commands.add_parser("adaptive-convergence")
    convergence.add_argument("--output-dir", type=Path, required=True)
    convergence.add_argument("--base-subdivision", type=int, default=7)
    convergence.add_argument(
        "--target-subdivisions", type=int, nargs=2, default=(9, 10)
    )
    convergence.add_argument("--core-radius-deg", type=float, default=1.0)
    convergence.add_argument("--transition-width-deg", type=float, default=1.0)
    convergence.add_argument(
        "--material", choices=("etopo5", "natural-earth"), default="etopo5"
    )
    convergence.add_argument("--etopo5-path", type=Path)
    convergence.add_argument("--device", default="auto")
    convergence.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    convergence.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=True
    )
    convergence.add_argument("--torch-compile-chunk-size", type=int, default=8)
    convergence.add_argument(
        "--stop-after-center", type=float, default=PAPER_FIGURE_7_DURATION_S
    )
    convergence.add_argument("--synchronize-every", type=int, default=256)
    convergence.add_argument("--sample-every", type=int, default=32)

    analyze_adaptive = commands.add_parser("analyze-adaptive")
    analyze_adaptive.add_argument("--input-dir", type=Path, required=True)
    analyze_adaptive.add_argument("--summary", type=Path, required=True)
    analyze_adaptive.add_argument("--figure", type=Path)
    analyze_adaptive.add_argument("--coarse-target", type=int, default=9)
    analyze_adaptive.add_argument("--fine-target", type=int, default=10)
    analyze_adaptive.add_argument(
        "--relative-l2-threshold",
        type=float,
        default=0.05,
        help="screening threshold applied to all six relative-L2 changes",
    )
    return parser


def _load_validation_traces(path: Path) -> ValidationTraces:
    with np.load(path) as values:
        return ValidationTraces(
            time_steps=values["time_steps"].astype(np.int64),
            time_s=values["time_s"].astype(np.float64),
            er_v_m=values["er_v_m"].astype(np.float64),
            labels=tuple(str(value) for value in values["labels"]),
        )


def _run_figures_5_6(args: argparse.Namespace) -> int:
    traces = _load_validation_traces(args.traces)
    curves = compute_attenuation(traces)
    figure_5 = render_figure_5(traces, args.output_dir / "figure-5.png")
    figure_6 = render_figure_6(curves, args.output_dir / "figure-6.png")
    print(f"figure_5={figure_5}")
    print(f"figure_6={figure_6}")
    return 0


def _run_radar(args: argparse.Namespace) -> int:
    if args.stop_after_center <= 0.0:
        raise SystemExit("--stop-after-center must be positive")
    if args.material == "etopo5" and args.etopo5_path is None:
        raise SystemExit("--etopo5-path is required with --material etopo5")
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be positive")
    if args.mesh_coordinates is not None and args.mesh_optimization_steps:
        raise SystemExit(
            "--mesh-coordinates cannot be combined with --mesh-optimization-steps"
        )
    optimized_mesh = None
    if args.mesh_coordinates is not None:
        try:
            optimized_mesh, _ = load_optimized_mesh(
                args.mesh_coordinates,
                expected_subdivision=args.subdivision,
                expected_orientation=args.mesh_orientation,
            )
        except (OSError, KeyError, ValueError) as error:
            raise SystemExit(str(error)) from error
    source_azimuths = {
        "both": (0.0, 90.0),
        "north": (0.0,),
        "east": (90.0,),
        "difference": (0.0, 270.0),
    }[args.source_basis]
    simulation = create_radar_simulation(
        include_oil=args.case == "anomaly",
        subdivision=args.subdivision,
        material_model=args.material,
        etopo5_path=args.etopo5_path,
        device=args.device,
        dtype=args.dtype,
        compile_step=args.torch_compile,
        compile_chunk_size=args.torch_compile_chunk_size,
        source_center_s=args.source_center,
        courant_factor=args.courant,
        source_edge_assignment=args.source_edge_assignment,
        tangential_interface_mode=args.tangential_interface,
        tangential_material_support=args.tangential_support,
        source_altitude_m=args.source_altitude_m,
        source_azimuths_deg=source_azimuths,
        include_shield=args.shield,
        shield_radius_m=1_000.0 * args.shield_radius_km,
        mesh_orientation=args.mesh_orientation,
        mesh_optimization_steps=args.mesh_optimization_steps,
        mesh=optimized_mesh,
        geometry_mode=args.geometry_mode,
        vertical_reference=args.vertical_reference,
        horizontal_anomaly_mode=args.horizontal_anomaly,
        deep_lithosphere_resistivity_ohm_m=(
            args.deep_lithosphere_resistivity_ohm_m
        ),
        upper_crust_resistivity_ohm_m=args.upper_crust_resistivity_ohm_m,
        asthenosphere_resistivity_ohm_m=args.asthenosphere_resistivity_ohm_m,
        lithosphere_profile=args.lithosphere_profile,
        ionosphere_model=args.ionosphere,
        subsolar_latitude_deg=args.subsolar_latitude_deg,
        subsolar_longitude_deg=args.subsolar_longitude_deg,
    )
    steps = int(
        np.ceil((args.source_center + args.stop_after_center) / simulation.time_step_s)
    )
    print(
        f"case={args.case} grid={simulation.mesh.n_vertices:,}x"
        f"{len(simulation.radial_steps_m)} runtime={simulation.runtime} "
        f"device={simulation.device} dtype={simulation.dtype_name} "
        f"orientation={args.mesh_orientation} "
        f"mesh_optimization_steps={args.mesh_optimization_steps} "
        f"mesh_coordinates={args.mesh_coordinates or 'generated'} "
        f"geometry={args.geometry_mode} "
        f"interface={args.tangential_interface} "
        f"support={args.tangential_support} "
        f"source={args.source_basis}@{simulation.source.altitude_m:g}m "
        f"receiver={args.receiver_support} "
        f"vertical_reference={args.vertical_reference} "
        f"horizontal_anomaly={args.horizontal_anomaly} "
        f"lithosphere_profile={args.lithosphere_profile} "
        f"ionosphere={args.ionosphere} "
        f"subsolar={args.subsolar_latitude_deg:g},"
        f"{args.subsolar_longitude_deg:g} "
        f"upper_crust_resistivity_ohm_m="
        f"{args.upper_crust_resistivity_ohm_m:g} "
        f"asthenosphere_resistivity_ohm_m="
        f"{args.asthenosphere_resistivity_ohm_m:g} "
        "deep_lithosphere_resistivity_ohm_m="
        f"{args.deep_lithosphere_resistivity_ohm_m:g} "
        f"shield={args.shield_radius_km:g}km/{args.shield} "
        f"dt={simulation.time_step_s:.9e}s steps={steps:,}",
        flush=True,
    )
    started = time.perf_counter()
    traces = record_radar_traces(
        simulation,
        steps=steps,
        case=args.case,
        synchronize_every=args.synchronize_every,
        receiver_support=args.receiver_support,
        sample_every=args.sample_every,
    )
    output = save_radar_traces(traces, args.output)
    print(f"elapsed_s={time.perf_counter() - started:.3f} output={output}", flush=True)
    return 0


def _analyze_radar(args: argparse.Namespace) -> int:
    reference = load_radar_traces(args.reference)
    anomaly = load_radar_traces(args.anomaly)
    curves = compute_radar_perturbation(
        reference,
        anomaly,
        normalization=args.normalization,
        ht_definition=args.ht_definition,
    )
    figure = render_figure_7(curves, args.figure)
    print(
        f"figure={figure} normalization={args.normalization} "
        f"ht_definition={args.ht_definition}"
    )
    metrics = radar_metrics(curves)
    metrics.update(radar_field_metrics(reference, anomaly, curves))
    for name, value in metrics.items():
        print(f"{name}={value:.9g}")
    return 0


def _run_adaptive_convergence(args: argparse.Namespace) -> int:
    coarse_target, fine_target = args.target_subdivisions
    if not args.base_subdivision < coarse_target < fine_target:
        raise SystemExit(
            "--target-subdivisions must be increasing and above --base-subdivision"
        )
    if args.stop_after_center <= 0.0:
        raise SystemExit("--stop-after-center must be positive")
    if args.material == "etopo5" and args.etopo5_path is None:
        raise SystemExit("--etopo5-path is required with --material etopo5")
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces_by_level = {}
    level_metadata = []

    for target in (coarse_target, fine_target):
        mesh = build_paper_adaptive_mesh(
            target,
            base_subdivision=args.base_subdivision,
            core_radius_deg=args.core_radius_deg,
            transition_width_deg=args.transition_width_deg,
        )
        pair = {}
        level_started = time.perf_counter()
        for case in ("reference", "anomaly"):
            simulation = create_radar_simulation(
                include_oil=case == "anomaly",
                subdivision=args.base_subdivision,
                material_model=args.material,
                etopo5_path=args.etopo5_path,
                        device=args.device,
                dtype=args.dtype,
                compile_step=args.torch_compile,
                compile_chunk_size=args.torch_compile_chunk_size,
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
                f"target=s{target} case={case} faces={mesh.n_faces:,} "
                f"dt={simulation.time_step_s:.9e}s steps={steps:,} "
                f"device={simulation.device}",
                flush=True,
            )
            traces = record_radar_traces(
                simulation,
                steps=steps,
                case=case,
                synchronize_every=args.synchronize_every,
                sample_every=args.sample_every,
            )
            pair[case] = traces
            del simulation
            gc.collect()
        for case, traces in pair.items():
            path = save_radar_traces(
                traces, args.output_dir / f"s{target}-{case}.npz"
            )
            print(f"output={path}", flush=True)
        traces_by_level[target] = pair
        level_metadata.append(
            {
                "target_subdivision": target,
                "vertices": mesh.n_vertices,
                "edges": mesh.n_edges,
                "faces": mesh.n_faces,
                "face_level_counts": {
                    str(level): int(np.count_nonzero(mesh.face_levels == level))
                    for level in np.unique(mesh.face_levels)
                },
                "refinement_spec": mesh.refinement_spec,
                "elapsed_s": time.perf_counter() - level_started,
            }
        )
        del mesh
        gc.collect()

    convergence = compare_radar_resolution_pairs(
        traces_by_level[coarse_target]["reference"],
        traces_by_level[coarse_target]["anomaly"],
        traces_by_level[fine_target]["reference"],
        traces_by_level[fine_target]["anomaly"],
        coarse_target_subdivision=coarse_target,
        fine_target_subdivision=fine_target,
        relative_stop_s=args.stop_after_center,
    )
    summary = args.output_dir / "convergence.json"
    summary.write_text(
        json.dumps(
            {"levels": level_metadata, "comparison": asdict(convergence)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary={summary}", flush=True)
    return 0


def _analyze_adaptive(args: argparse.Namespace) -> int:
    if args.coarse_target >= args.fine_target:
        raise SystemExit("--coarse-target must be below --fine-target")
    if not 0.0 < args.relative_l2_threshold < 1.0:
        raise SystemExit("--relative-l2-threshold must be in (0, 1)")

    def load(target: int, case: str):
        return load_radar_traces(args.input_dir / f"s{target}-{case}.npz")

    coarse_reference = load(args.coarse_target, "reference")
    coarse_anomaly = load(args.coarse_target, "anomaly")
    fine_reference = load(args.fine_target, "reference")
    fine_anomaly = load(args.fine_target, "anomaly")
    signatures = [
        json.loads(traces.run_signature)
        for traces in (
            coarse_reference,
            coarse_anomaly,
            fine_reference,
            fine_anomaly,
        )
    ]
    dtypes = {signature["dtype"] for signature in signatures}
    runtimes = {signature["runtime"] for signature in signatures}
    if len(dtypes) != 1 or len(runtimes) != 1:
        raise SystemExit("adaptive traces must share one runtime and dtype")

    convergence = compare_radar_resolution_pairs(
        coarse_reference,
        coarse_anomaly,
        fine_reference,
        fine_anomaly,
        coarse_target_subdivision=args.coarse_target,
        fine_target_subdivision=args.fine_target,
    )
    convergence_values = {
        name: value
        for name, value in asdict(convergence).items()
        if name.endswith("_relative_l2")
    }
    maximum_relative_l2 = max(convergence_values.values())
    fine_metrics = {}
    pointwise_curves = None
    for normalization in ("pointwise", "peak"):
        curves = compute_radar_perturbation(
            fine_reference,
            fine_anomaly,
            normalization=normalization,
            ht_definition="vector-difference",
        )
        metrics = radar_metrics(curves)
        metrics.update(radar_field_metrics(fine_reference, fine_anomaly, curves))
        fine_metrics[normalization] = metrics
        if normalization == "pointwise":
            pointwise_curves = curves

    payload = {
        "format_version": 1,
        "screening": {
            "runtime": runtimes.pop(),
            "dtype": dtypes.pop(),
            "relative_l2_threshold": args.relative_l2_threshold,
            "maximum_relative_l2": maximum_relative_l2,
            "converged": bool(
                np.isfinite(maximum_relative_l2)
                and maximum_relative_l2 <= args.relative_l2_threshold
            ),
        },
        "comparison": asdict(convergence),
        "fine_metrics": fine_metrics,
        "provenance": {
            f"s{target}_{case}": {
                "path": str(args.input_dir / f"s{target}-{case}.npz"),
                "git_revision": signatures[index]["git_revision"],
                "mesh_vertices_sha256": signatures[index][
                    "mesh_vertices_sha256"
                ],
                "mesh_faces_sha256": signatures[index]["mesh_faces_sha256"],
                "time_step_s": signatures[index]["time_step_s"],
            }
            for index, (target, case) in enumerate(
                (
                    (args.coarse_target, "reference"),
                    (args.coarse_target, "anomaly"),
                    (args.fine_target, "reference"),
                    (args.fine_target, "anomaly"),
                )
            )
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.summary)
    print(
        f"summary={args.summary} max_relative_l2={maximum_relative_l2:.9g} "
        f"converged={payload['screening']['converged']}",
        flush=True,
    )
    if args.figure is not None:
        assert pointwise_curves is not None
        figure = render_figure_7(pointwise_curves, args.figure)
        print(f"figure={figure}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "figures-5-6":
        return _run_figures_5_6(args)
    if args.command == "radar-run":
        return _run_radar(args)
    if args.command == "analyze-radar":
        return _analyze_radar(args)
    if args.command == "analyze-adaptive":
        return _analyze_adaptive(args)
    return _run_adaptive_convergence(args)


if __name__ == "__main__":
    raise SystemExit(main())

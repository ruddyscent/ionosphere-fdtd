"""Command-line rendering for geodesic FDTD maps, sections, traces, and 3-D views."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._torch_runtime import BackendUnavailableError
from .checkpoint import CheckpointError
from .cli_common import DefaultsHelpFormatter, add_version_argument
from .cli_config import (
    add_config_argument,
    apply_toml_defaults,
    clear_explicit_append_defaults,
    explicit_subcommand,
    load_toml_from_argv,
    reject_legacy_backend_argument,
    subparser,
    table,
    validate_nested_tables,
    validate_root_sections,
)
from .solver import GeodesicFDTD, SimulationConfig
from .sources import (
    GWANGJU_LATITUDE_DEG,
    GWANGJU_LONGITUDE_DEG,
    GaussianCurrent,
)
from .visualization import (
    Receiver,
    animate_surface_field,
    plot_mesh_3d,
    plot_radial_section,
    plot_receiver_traces,
    plot_surface_field,
    record_receiver_traces,
    run_live_surface,
    sample_radial_section,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=DefaultsHelpFormatter
    )
    add_config_argument(parser)
    add_version_argument(parser)
    parser.add_argument(
        "--resume",
        type=Path,
        help="load model and fields from an NPZ checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device: cpu, auto, mps, cuda, cuda:N, or gpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float64"),
        default="float64",
        help="PyTorch field precision",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compile chunked PyTorch field steps for long-running simulations",
    )
    parser.add_argument(
        "--torch-compile-chunk-size",
        type=int,
        default=8,
        help="number of time steps captured in each compiled graph",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="set PyTorch CPU intra-op threads (small grids often prefer 1)",
    )
    parser.add_argument(
        "--subdivision",
        type=int,
        default=2,
        choices=range(0, 8),
        help="recursive surface-mesh refinement level",
    )
    parser.add_argument(
        "--radial-cells",
        type=int,
        default=24,
        help="number of radial intervals",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="warm-up steps for a new model; additional steps after a checkpoint "
        "(default: 100 new, 0 resumed)",
    )
    parser.add_argument(
        "--source-current",
        type=float,
        default=1.0e6,
        help="peak source current in amperes",
    )
    parser.add_argument(
        "--source-frequency",
        type=float,
        default=0.0,
        help="source carrier frequency in hertz; zero disables the carrier",
    )
    parser.add_argument(
        "--source-latitude",
        type=float,
        default=GWANGJU_LATITUDE_DEG,
        help="source geodetic latitude in degrees",
    )
    parser.add_argument(
        "--source-longitude",
        type=float,
        default=GWANGJU_LONGITUDE_DEG,
        help="source longitude in degrees east",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    surface = subparsers.add_parser("surface", help="render a projected field map")
    surface.add_argument("--component", choices=("er", "hr"), default="er")
    surface.add_argument("--altitude-km", type=float, default=0.0)
    surface.add_argument("--projection", default="mollweide")
    surface.add_argument("--scale", choices=("linear", "symlog"), default="linear")
    surface.add_argument("--color-limit", type=float)
    surface.add_argument(
        "--coastlines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlay Natural Earth coastlines (may download data on first use)",
    )
    surface.add_argument("--output", type=Path)

    section = subparsers.add_parser("section", help="render a distance-height section")
    section.add_argument(
        "--start-latitude", type=float, default=GWANGJU_LATITUDE_DEG
    )
    section.add_argument(
        "--start-longitude", type=float, default=GWANGJU_LONGITUDE_DEG
    )
    section.add_argument(
        "--end-latitude", type=float, default=-GWANGJU_LATITUDE_DEG
    )
    section.add_argument(
        "--end-longitude", type=float, default=GWANGJU_LONGITUDE_DEG - 180.0
    )
    section.add_argument("--samples", type=int, default=241)
    section.add_argument("--scale", choices=("linear", "symlog"), default="linear")
    section.add_argument("--color-limit", type=float)
    section.add_argument("--output", type=Path)

    mesh = subparsers.add_parser("mesh", help="render a 3-D geodesic surface")
    mesh.add_argument("--component", choices=("topology", "er", "hr"), default="topology")
    mesh.add_argument("--altitude-km", type=float, default=0.0)
    mesh.add_argument("--color-limit", type=float)
    mesh.add_argument(
        "--earth-texture", action=argparse.BooleanOptionalAction, default=True
    )
    mesh.add_argument("--field-opacity", type=float, default=0.82)
    mesh.add_argument("--output", type=Path)

    animation = subparsers.add_parser("animate", help="write a GIF or MP4")
    animation.add_argument("--component", choices=("er", "hr"), default="er")
    animation.add_argument("--altitude-km", type=float, default=0.0)
    animation.add_argument("--frames", type=int, default=120)
    animation.add_argument("--steps-per-frame", type=int, default=10)
    animation.add_argument("--fps", type=int, default=24)
    animation.add_argument("--color-limit", type=float)
    animation.add_argument(
        "--earth-texture", action=argparse.BooleanOptionalAction, default=True
    )
    animation.add_argument("--field-opacity", type=float, default=0.82)
    animation.add_argument(
        "--show-edges", action=argparse.BooleanOptionalAction, default=True
    )
    animation.add_argument("--output", type=Path)

    live = subparsers.add_parser(
        "live", help="advance the solver in an interactive 3-D window"
    )
    live.add_argument("--component", choices=("er", "hr"), default="er")
    live.add_argument("--altitude-km", type=float, default=0.0)
    live.add_argument("--steps-per-frame", type=int, default=10)
    live.add_argument("--fps", type=int, default=20)
    live.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop calculation after this many frames; 0 runs until the window closes",
    )
    live.add_argument("--color-limit", type=float)
    live.add_argument(
        "--earth-texture", action=argparse.BooleanOptionalAction, default=True
    )
    live.add_argument("--field-opacity", type=float, default=0.82)
    live.add_argument(
        "--show-edges", action=argparse.BooleanOptionalAction, default=True
    )

    traces = subparsers.add_parser("traces", help="render receiver time series")
    traces.add_argument("--trace-steps", type=int, default=1000)
    traces.add_argument("--sample-every", type=int, default=10)
    traces.add_argument(
        "--receiver",
        nargs=3,
        action="append",
        metavar=("LAT", "LON", "ALT_KM"),
        type=float,
    )
    traces.add_argument("--output", type=Path)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    reject_legacy_backend_argument(argv)
    parser = _parser()
    _, document = load_toml_from_argv(argv)
    validate_root_sections(document, allowed={"ionosphere", "visualization"})
    values = table(document, ("visualization",))
    commands = {"surface", "section", "mesh", "animate", "live", "traces"}
    validate_nested_tables(values, allowed=commands, section="visualization")
    apply_toml_defaults(parser, values, section="visualization")
    for configured_command in commands:
        command_values = table(document, ("visualization", configured_command))
        apply_toml_defaults(
            subparser(parser, configured_command),
            command_values,
            section=f"visualization.{configured_command}",
        )
    command = explicit_subcommand(argv, commands)
    if command is not None:
        command_parser = subparser(parser, command)
        clear_explicit_append_defaults(command_parser, argv)
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 0 if args.resume is not None else 100
    if args.command != "live" and args.output is None:
        parser.error(f"{args.command} requires --output or a TOML output value")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if args.torch_threads is not None and args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    if args.torch_compile_chunk_size < 1:
        raise SystemExit("--torch-compile-chunk-size must be positive")
    try:
        if args.resume is not None:
            simulation = GeodesicFDTD.load_checkpoint(
                args.resume,
                device=args.device,
                dtype=None if args.dtype == "auto" else args.dtype,
                compile_step=args.torch_compile,
                compile_chunk_size=args.torch_compile_chunk_size,
                torch_threads=args.torch_threads,
            )
        else:
            simulation = GeodesicFDTD(
                config=SimulationConfig(
                    subdivision=args.subdivision, radial_cells=args.radial_cells
                ),
                source=GaussianCurrent(
                    latitude_deg=args.source_latitude,
                    longitude_deg=args.source_longitude,
                    peak_current_a=args.source_current,
                    carrier_frequency_hz=args.source_frequency,
                ),
                device=args.device,
                dtype=args.dtype,
                compile_step=args.torch_compile,
                compile_chunk_size=args.torch_compile_chunk_size,
                torch_threads=args.torch_threads,
            )
    except (BackendUnavailableError, CheckpointError, OSError) as error:
        raise SystemExit(str(error)) from error
    thread_text = (
        f" threads={simulation.threads}" if simulation.threads is not None else ""
    )
    print(
        f"runtime={simulation.runtime} device={simulation.device} "
        f"dtype={simulation.dtype_name}{thread_text} "
        f"compiled={simulation.compiled} "
        f"compile_chunk_size={simulation.compile_chunk_size}"
    )
    if args.resume is not None:
        print(f"checkpoint={args.resume} loaded_step={simulation.steps}")
    simulation.step(args.steps)
    output = getattr(args, "output", None)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "surface":
        figure, _, _ = plot_surface_field(
            simulation,
            args.component,
            altitude_m=1.0e3 * args.altitude_km,
            projection=args.projection,
            scale=args.scale,
            color_limit=args.color_limit,
            coastlines=args.coastlines,
        )
        figure.savefig(args.output, dpi=180)
        _close_figure(figure)
    elif args.command == "section":
        radial_section = sample_radial_section(
            simulation,
            args.start_latitude,
            args.start_longitude,
            args.end_latitude,
            args.end_longitude,
            samples=args.samples,
        )
        figure, _, _ = plot_radial_section(
            radial_section, scale=args.scale, color_limit=args.color_limit
        )
        figure.savefig(args.output, dpi=180)
        _close_figure(figure)
    elif args.command == "mesh":
        plot_mesh_3d(
            simulation,
            args.component,
            altitude_m=1.0e3 * args.altitude_km,
            color_limit=args.color_limit,
            earth_texture=args.earth_texture,
            field_opacity=args.field_opacity,
            screenshot=args.output,
        )
    elif args.command == "animate":
        animate_surface_field(
            simulation,
            args.output,
            component=args.component,
            altitude_m=1.0e3 * args.altitude_km,
            frames=args.frames,
            steps_per_frame=args.steps_per_frame,
            frames_per_second=args.fps,
            color_limit=args.color_limit,
            earth_texture=args.earth_texture,
            field_opacity=args.field_opacity,
            show_edges=args.show_edges,
        )
    elif args.command == "live":
        if args.frames < 0:
            raise SystemExit("--frames must be non-negative")
        completed_frames = run_live_surface(
            simulation,
            args.component,
            altitude_m=1.0e3 * args.altitude_km,
            steps_per_frame=args.steps_per_frame,
            frames_per_second=args.fps,
            max_frames=args.frames or None,
            color_limit=args.color_limit,
            show_edges=args.show_edges,
            earth_texture=args.earth_texture,
            field_opacity=args.field_opacity,
        )
        print(
            f"completed {completed_frames} live frames; "
            f"simulation step {simulation.steps}"
        )
    else:
        receiver_specs = args.receiver or [
            (35.6762, 139.6503, 0.0),
            (21.3069, -157.8583, 0.0),
        ]
        receivers = [
            Receiver(latitude, longitude, 1.0e3 * altitude)
            for latitude, longitude, altitude in receiver_specs
        ]
        receiver_traces = record_receiver_traces(
            simulation,
            receivers,
            args.trace_steps,
            sample_every=args.sample_every,
        )
        figure, _ = plot_receiver_traces(receiver_traces)
        figure.savefig(args.output, dpi=180)
        _close_figure(figure)
    if output is not None:
        print(output)
    return 0


def _close_figure(figure: object) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)

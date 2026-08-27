# Command-Line Reference

## Simulation runner

`ionosphere` advances the solver and periodically prints scalar diagnostics.
List every option with:

```bash
uv run ionosphere --help
```

The principal option groups are:

| Group | Options |
|---|---|
| Work | `--steps`, `--dry-run`, `--report-every`, `--resume`, `--checkpoint`, `--checkpoint-every` |
| Grid | `--subdivision`, `--radial-cells`, `--surface-step`, `--courant` |
| Runtime | `--device`, `--dtype`, `--torch-compile`, `--torch-compile-chunk-size`, `--torch-threads` |
| Source | `--source-current`, `--source-length`, `--source-frequency`, `--source-center`, `--source-width`, `--source-latitude`, `--source-longitude` |
| Anomaly | `--oil-anomaly`, `--anomaly-radius-km` |

This installed runner intentionally covers the compact data-free model. Use the
Python API for conforming adaptive surface meshes, balanced radial refinement,
gridded or mesh-native materials, surface impedance, magnetized plasma, and
generic two-rank construction. Paper-specific adaptive and distributed runners
are source-only modules under `verification/` and are not installed console
scripts.

`--backend` was removed. PyTorch is the only compute runtime; select hardware
with `--device` and precision with `--dtype`. The TOML `backend` key is
also removed and is rejected with migration guidance. See the
[0.2.0 migration table](pytorch-only-migration.md#old-to-new-reference).

Run a preflight before a long job:

```bash
uv run ionosphere --config run.toml --dry-run
```

The command constructs and validates the selected mesh, material, source,
device, precision, and time step, then reports exact field memory
without advancing a field step or writing a configured checkpoint. Use
`--version` to report the installed package version. Boolean options accept
both forms, such as `--torch-compile` and `--no-torch-compile`, so command-line
values can override either TOML boolean value.

## TOML configuration files

Both command-line applications accept `--config PATH`. Configuration values
become parser defaults, so an explicitly supplied command-line option always
wins:

```bash
cp configs/ionosphere.example.toml run.toml
uv run ionosphere --config run.toml

# Reuse the file but override only this run's device and step count.
uv run ionosphere --config run.toml --device cpu --steps 100
```

Simulation-runner values belong in `[ionosphere]`. TOML keys use the argparse
destination spelling: replace option hyphens with underscores. For example,
`--radial-cells` becomes `radial_cells`, `--torch-compile` becomes
`torch_compile`, and flags use TOML booleans.

```toml
[ionosphere]
steps = 20000
subdivision = 5
radial_cells = 40
device = "cuda:0"
dtype = "float32"
torch_compile = true
torch_compile_chunk_size = 32
source_frequency = 20.0
checkpoint = "artifacts/runs/model.npz"
checkpoint_every = 5000
```

The visualization runner reads shared simulation defaults from
`[visualization]` and render-specific values from
`[visualization.COMMAND]`. Place `--config` before the command:

```toml
[visualization]
subdivision = 4
radial_cells = 40
steps = 100

[visualization.surface]
component = "er"
scale = "symlog"
coastlines = true
output = "artifacts/figures/surface.png"
```

```bash
uv run --extra visualization ionosphere-visualize \
  --config run.toml surface
```

The tables for `surface`, `section`, `mesh`, `animate`, `live`, and `traces`
accept the same names as their command-specific options. Repeatable values use
arrays; for example, `receiver` is an array of `[latitude, longitude,
altitude_km]` arrays. Relative paths are interpreted from the command's current
working directory. Unknown keys, invalid types, unsupported choices, malformed
TOML, and missing files terminate with a diagnostic instead of being ignored.
See
[`configs/ionosphere.example.toml`](../../configs/ionosphere.example.toml) for
a CPU-safe starting point. The longer compiled CUDA example is kept separately
in
[`configs/ionosphere.research.toml`](../../configs/ionosphere.research.toml).
A file may contain both application tables; each command reads only its own
table. A visualization command must still be selected on the command line
because the configuration does not choose a subcommand.

`--surface-step SPACING_M` adds regularly spaced radial nodes within 5 km of
sea level. Because this creates abrupt transitions to the coarse grid, the CLI
selects the explicitly permitted first-order transition policy. Use the Python
API when a smoothly graded custom grid is required.

The built-in oil anomaly is centered near Alaska, extends from 2 km to 0.5 km
below sea level, and multiplies lithosphere conductivity by 0.1. A 40 km radius
is too small for coarse demonstration grids; the runner warns when no electric
sample intersects its support. For a visibly resolved smoke experiment:

```bash
uv run ionosphere \
  --subdivision 3 --radial-cells 40 --surface-step 1250 \
  --oil-anomaly --anomaly-radius-km 1200 \
  --source-frequency 20 --steps 1000
```

The enlarged anomaly is a numerical demonstration, not the published radar
geometry.

## Checkpoints and restart

Write a final checkpoint and refresh it every 1,000 completed steps:

```bash
uv run ionosphere \
  --steps 10000 --checkpoint run.npz --checkpoint-every 1000
```

Resume the embedded model and field state for 5,000 additional steps:

```bash
uv run ionosphere --resume run.npz --steps 5000 --checkpoint run.npz
```

`--steps` always means additional steps. On resume, the checkpoint owns the
grid, material, source, time step, and current step count. Historical v1–v4
checkpoints can be resumed on a caller-selected Torch device, for example
`--device cuda`; their stored legacy backend metadata is advisory and does
not select a removed runtime. With `--dtype auto`, restart preserves the
stored dtype; an explicit `--dtype` converts fields to that precision.

Checkpoint updates are atomic: the completed temporary archive replaces the
destination only after it has been written successfully. `--checkpoint-every`
requires `--checkpoint`, and the final state is always written even when the
requested step count is not an exact checkpoint interval.

## Visualization runner

`ionosphere-visualize` uses global simulation options followed by one required
subcommand:

```text
ionosphere-visualize [--config PATH] [--resume CHECKPOINT] [simulation options] COMMAND [render options]
```

| Command | Output |
|---|---|
| `surface` | Projected `Er` or `Hr` map |
| `section` | Great-circle distance–height `Er` section |
| `mesh` | Static 3-D topology or field surface |
| `animate` | GIF or MP4 field animation |
| `live` | Interactive advancing 3-D field |
| `traces` | One or more receiver time series |

Global `--steps` are warm-up steps performed before rendering or trace
recording. A new model defaults to 100 warm-up steps. A resumed model defaults
to zero additional steps, so a surface, section, or mesh command renders the
saved state unless `--steps` is explicitly supplied. Options after the
subcommand belong to that output mode. Show
subcommand-specific help by placing `--help` after its name:

```bash
uv run --extra visualization ionosphere-visualize surface --help
uv run --extra visualization ionosphere-visualize animate --help
```

Angles are in degrees. Receiver altitude and visualization `--altitude-km`
values are in kilometres; source coordinates use the model's configured
default altitude because the visualization runner exposes only source latitude
and longitude.

Render the exact model and field state saved by a simulation:

```bash
uv run --extra visualization ionosphere-visualize \
  --resume artifacts/runs/demo.npz \
  surface --component er --scale symlog --output surface.png
```

The checkpoint owns the mesh, radial grid, material, source, time step, and
current field state. Device, dtype, and compilation options may be changed
while loading. For `traces`, `--trace-steps` advances from the saved
state and records new samples; checkpoints do not contain historical receiver
traces.

## Exit behavior

Invalid grids, unsupported device/dtype combinations, unstable time
steps, and unresolved required arguments terminate with a nonzero status and a
diagnostic message. Normal runs print the selected mesh, runtime, time step,
memory use, and field maxima.

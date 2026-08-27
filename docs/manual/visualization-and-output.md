# Visualization and Output

Install the visualization extra before using `ionosphere-visualize`:

```bash
uv sync --extra visualization
```

## Surface maps

Render an existing checkpoint without rerunning its model:

```bash
uv run --extra visualization ionosphere-visualize \
  --resume artifacts/runs/demo.npz \
  surface --component er --scale symlog --output surface.png
```

With `--resume`, the default warm-up is zero steps and the saved field state is
rendered directly. An explicit global `--steps N` advances the restored solver
by `N` additional steps first.

To construct a standalone demonstration instead, omit `--resume`:

```bash
uv run --extra visualization ionosphere-visualize \
  --subdivision 2 --radial-cells 24 --steps 1200 \
  surface --component er --scale symlog --output surface.png
```

Surface maps interpolate display values onto a regular longitude/latitude grid;
the solver fields are not modified. The requested altitude selects the nearest
`Er` radial node or `Hr` radial midpoint; it does not interpolate between
staggered planes. The rendered title reports the plane actually selected. `er`
and `hr` maps use symmetric color limits about zero.

## Radial sections

```bash
uv run --extra visualization ionosphere-visualize \
  --subdivision 2 --radial-cells 24 --steps 1200 \
  section \
  --start-latitude 35.1595 --start-longitude 126.8526 \
  --end-latitude -35.1595 --end-longitude -53.1474 \
  --output section.png
```

## Mesh and interactive view

```bash
uv run --extra visualization ionosphere-visualize \
  --subdivision 2 --steps 0 \
  mesh --component topology --output mesh.png

uv run --extra visualization ionosphere-visualize \
  --subdivision 2 --radial-cells 24 --steps 0 \
  live --component er --steps-per-frame 10 --fps 20
```

In the live view, drag to rotate, use the wheel to zoom, and press `q` or close
the window to stop. Use `--no-earth-texture`, `--no-show-edges`, and
`--field-opacity` to simplify the display.

## Animations

```bash
uv run --extra visualization ionosphere-visualize \
  --subdivision 2 --radial-cells 24 --steps 1200 \
  animate --frames 120 --steps-per-frame 10 --fps 24 \
  --color-limit 4 --output field.mp4
```

Use an explicit color limit when comparing multiple runs. A nonzero warm-up is
usually needed before the first frame.

## Receiver traces

```bash
uv run --extra visualization ionosphere-visualize \
  --subdivision 2 --steps 40 \
  traces --trace-steps 4000 --sample-every 10 \
  --receiver 35.6762 139.6503 0 \
  --receiver 21.3069 -157.8583 0 \
  --output traces.png
```

The visualization runner records `Er` at the nearest mesh vertex and nearest
radial node for each requested receiver. It does not interpolate to the exact
geographic coordinate or altitude, and each plotted sample is read as a host
scalar. This simple path is intended for inspection. For accelerator-resident
weighted sampling, construct explicit support indices and weights and use
`record_er_observations()` or `record_h_observations()`. Those methods return
backend-native trace buffers without detaching them. Call
`simulation.to_numpy(traces)` once at the terminal plotting or artifact
boundary; this is the explicit device synchronization and host-copy boundary.

## Python plotting API

```python
from ionosphere_fdtd import plot_surface_field, sample_radial_section

figure, axes, artist = plot_surface_field(
    simulation, "er", altitude_m=0.0, scale="symlog"
)
section = sample_radial_section(
    simulation, 35.1595, 126.8526, -35.1595, -53.1474
)
```

Call `simulation.to_numpy(field)` before passing backend-native arrays to other
host plotting or serialization libraries.

## Portable checkpoint files

The Python API stores complete restartable results in a compressed, versioned
NPZ file:

```python
simulation.step(10_000)
simulation.save_checkpoint("run.npz")

restored = GeodesicFDTD.load_checkpoint(
    "run.npz", device="cuda"
)
restored.step(5_000)
```

The current format is version 4, and the loader accepts legacy versions 1–4.
The archive contains JSON metadata, exact mesh topology and refinement
metadata, all four evolving fields, the simulation clock, configuration,
material, and source. Version 3 added surface-impedance ADE memory; version 4
adds the mesh-bound plasma model and every species-current ADE state. Loading
uses `allow_pickle=False`; checkpoint files never execute serialized Python
objects.

Checkpoint saving currently supports `EarthIonosphereMaterial` and optional
`GaussianCurrent` or `TangentialGaussianCurrent` sources. Other runtime
materials, including layered, spatial, gridded, and mesh-artifact inputs, are
rejected even when they contain no external callable. Preserve those input
artifacts and their provenance separately until a portable checkpoint schema
is defined for them.

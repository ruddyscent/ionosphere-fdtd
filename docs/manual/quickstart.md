# Quick Start

## Run the default model

```bash
uv run ionosphere --steps 200
```

The default uses PyTorch CPU, subdivision 2, 24 radial cells, a Gwangju vertical
Gaussian current, and the data-free Earth–ionosphere material. Progress output
includes simulation time and maximum electric and magnetic amplitudes.

Expected setup output begins with a 162-cell surface mesh, PyTorch on CPU,
`float64`, a field allocation of about 0.27 MiB, and a conservative time step.
The exact field maxima depend on the requested step count.

## Reuse run parameters

Copy the CPU-safe example TOML file when a run needs more than a few options:

```bash
cp configs/ionosphere.example.toml run.toml
uv run ionosphere --config run.toml
```

The starter uses the same subdivision, radial grid, runtime, source, and dtype
by restoring the simulation checkpoint for visualization. It writes a final
checkpoint to `artifacts/runs/demo.npz`. The separate
[`ionosphere.research.toml`](../../configs/ionosphere.research.toml) retains a
long compiled CUDA example; review its device, resolution, step count, and
checkpoint path before running it.

Validate the complete model without advancing fields or writing the configured
checkpoint:

```bash
uv run ionosphere --config run.toml --dry-run
```

Run the model, then render that exact saved state:

```bash
uv run ionosphere --config run.toml
uv run --extra visualization ionosphere-visualize \
  --config run.toml surface
```

The second command reads `artifacts/runs/demo.npz`; it does not construct a
look-alike model. A successful workflow creates:

```text
artifacts/
├── figures/demo-surface.png
└── runs/demo.npz
```

The image confirms that the numerical workflow is operating. The starter is a
data-free demonstration and is not an observational Earth prediction. Continue
with the [Learning Path](learning-path.md) before interpreting amplitudes or
travel times physically.

Edit values below `[ionosphere]` in `run.toml`. An option supplied directly on
the command line takes precedence, which makes one-off changes concise:

```bash
uv run ionosphere --config run.toml --steps 100 --device cpu
```

The same file can contain `[visualization]` and
`[visualization.COMMAND]` tables for `ionosphere-visualize`. See the
[TOML configuration reference](command-line-reference.md#toml-configuration-files)
for the table layout and validation rules.

## Select a grid

```bash
# Small smoke run
uv run ionosphere --subdivision 1 --radial-cells 16 --steps 100

# A 642-cell demonstration grid
uv run ionosphere --subdivision 3 --radial-cells 40 --steps 1000
```

Surface dual-cell counts follow

$$
N_{\mathrm{surface}}=10\,4^L+2,
$$

where $L$ is the subdivision level.

| Level | Dual cells | Approximate center spacing |
|---:|---:|---:|
| 0 | 12 | 7,054 km |
| 1 | 42 | 3,765 km |
| 2 | 162 | 1,910 km |
| 3 | 642 | 962 km |
| 4 | 2,562 | 482 km |
| 5 | 10,242 | 241 km |
| 6 | 40,962 | 121 km |
| 7 | 163,842 | 60 km |

The Simpson–Taflove paper-target grid is subdivision 7. Lower levels are useful
for smoke tests and controlled convergence sweeps, but they are not substitutes
for the paper's stated surface-cell count.

## Choose a device and precision

CPU and `float64` are the safe defaults. Select an accelerator and
`float32` explicitly after validating the model on CPU:

```bash
uv run ionosphere \
  --device cuda --dtype float32 --steps 200
```

`--device auto` probes CUDA, then MPS, then CPU, but it is not a performance
planner. MPS supports `float32` only; requesting `float64` fails instead of
silently changing precision.

For a long, fixed-shape run, evaluate compilation separately:

```bash
uv run ionosphere \
  --device cuda --torch-compile --steps 20000
```

## Use the Python API

```python
from ionosphere_fdtd import (
    EarthIonosphereMaterial,
    GaussianCurrent,
    GeodesicFDTD,
    SimulationConfig,
)

config = SimulationConfig(
    subdivision=2,
    radial_cells=24,
    minimum_altitude_m=-100_000.0,
    maximum_altitude_m=100_000.0,
    courant_factor=0.35,
)
simulation = GeodesicFDTD(
    config,
    material=EarthIonosphereMaterial(),
    source=GaussianCurrent(carrier_frequency_hz=20.0),
    dtype="float64",
)
simulation.step(1000)

print(simulation.diagnostics())
er = simulation.to_numpy(simulation.er)
```

`to_numpy()` is an explicit terminal boundary between Torch tensors and host
analysis code. It detaches the tensor from its autograd graph, moves it to CPU
memory, and returns a NumPy array; do not use that exported array to construct
a differentiable loss.

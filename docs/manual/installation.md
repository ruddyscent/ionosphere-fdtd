# Installation

## Requirements

- Python 3.11 or newer
- NumPy 2.0 or newer
- PyTorch 2.5 or newer
- Optional visualization, verification, TensorBoard, and test dependencies

The repository includes `uv.lock`; `uv` is the recommended environment and
dependency manager.

## Minimal source checkout

From a source checkout, install the required NumPy and PyTorch dependencies:

```bash
uv sync
```

Run commands through `uv run`:

```bash
uv run ionosphere --steps 10
uv run python -c "import ionosphere_fdtd; print(ionosphere_fdtd.__name__)"
```

Add plotting only when creating the first figure:

```bash
uv sync --extra visualization
```

## Optional dependency groups

| Extra | Packages and use |
|---|---|
| `test` | pytest runner and runtime unit-test dependencies |
| `visualization` | Matplotlib, Cartopy, PyVista, Pillow, and video output |
| `verification` | SciPy analytic roots and verification workflows |
| `tensorboard` | TensorBoard and TensorBoardX diagnostics |

Torch is part of the core installation. A minimal CPU deployment requires
Torch but does not require an accelerator build or visualization packages.
Use the official PyTorch installation selector when a particular CUDA or MPS
build is required; installing this project does not make unavailable hardware
or drivers usable.

For repository development and the complete test suite, install every
development extra:

```bash
uv sync --extra test --extra visualization --extra verification
```

## Editable pip installation

If `uv` is not available, use an isolated virtual environment and install the
checkout in editable mode:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,visualization,verification]'
```

## Verify the installation

```bash
uv run ionosphere --subdivision 1 --radial-cells 8 --steps 2
uv run --extra test --extra verification pytest -q
```

The first command should report the mesh, PyTorch runtime, CPU device,
`float64` dtype, time step, memory use, and
finite field maxima. The verification extra is required when collecting the
complete configured suite because its analytic-solution tests import SciPy.
Install the visualization extra as well to run its optional tests instead of
skipping them.

## Supported runtime platforms

| Device | Minimum software | Limits |
|---|---|---|
| CPU | Python 3.11, NumPy 2.0, PyTorch 2.5 | Default; `float32` and `float64` |
| NVIDIA CUDA | CUDA-enabled PyTorch 2.5 and a compatible driver/GPU | Explicit `--device cuda` or `cuda:N` |
| Apple MPS | MPS-enabled PyTorch 2.5 on supported macOS hardware | `float32` only; `float64` is rejected |

See the
[PyTorch-only migration and release guide](pytorch-only-migration.md) when
upgrading from a release that exposed a NumPy compute backend.

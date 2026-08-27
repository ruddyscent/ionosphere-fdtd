# Installation

## Requirements

- Python 3.11 or newer
- NumPy 2.0 or newer
- Optional PyTorch, visualization, verification, and test dependencies

The repository includes `uv.lock`; `uv` is the recommended environment and
dependency manager.

## Minimal source checkout

From a source checkout, install only the NumPy runtime needed by the first CPU
simulation:

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
| `pytorch` | PyTorch CPU, CUDA, and MPS backends |
| `visualization` | Matplotlib, Cartopy, PyVista, Pillow, and video output |
| `verification` | SciPy analytic roots and verification workflows |
| `tensorboard` | TensorBoard and TensorBoardX diagnostics |

Install only what a deployment needs. A minimal NumPy CPU installation does
not require PyTorch or visualization packages.

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
python -m pip install -e '.[test,visualization,pytorch,verification]'
```

## Verify the installation

```bash
uv run ionosphere --subdivision 1 --radial-cells 8 --steps 2
uv run --extra test --extra verification pytest -q
```

The first command should report the mesh, backend, time step, memory use, and
finite field maxima. The verification extra is required when collecting the
complete configured suite because its analytic-solution tests import SciPy.
Install the PyTorch and visualization extras as well to run their optional
tests instead of skipping them.

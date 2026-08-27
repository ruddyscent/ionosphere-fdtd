# Troubleshooting

## Device unavailable

PyTorch is a required dependency. Verify whether the installed build exposes
the requested accelerator:

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

CUDA requires a CUDA-enabled PyTorch build and driver. MPS requires macOS,
Apple silicon or a supported AMD GPU, and an MPS-enabled PyTorch build.

## MPS rejects `float64`

Use `--dtype float32`. The solver intentionally rejects `float64` on MPS rather
than silently changing precision.

## Time step exceeds the CFL limit

Remove the explicit `time_step_s`, reduce it, or reduce `courant_factor`. The
limit depends on the smallest surface/radial geometry and the fastest sampled
material wave speed.

## Source frequency is above Nyquist

Reduce `carrier_frequency_hz` or refine the grid so the stable time step is
smaller. The required condition is

$$
f_{\mathrm{carrier}}<\frac{1}{2\Delta t}.
$$

## An anomaly has no effect

The CLI warning means no electric sample intersects the requested anomaly.
Increase surface subdivision, refine radial spacing, enlarge the anomaly for a
demonstration, or use conservative material support where scientifically
appropriate.

## PyTorch CPU is slower than a historical NumPy result

Small grids are sensitive to framework and thread overhead. The historical
s2/r16 `float32` run measured PyTorch CPU at 0.60 times the removed NumPy
runtime. Try `--torch-threads 1`, compare the standardized benchmark, and
size the production workload explicitly. The old NumPy compute runtime is not
available as a fallback. More threads do not guarantee better performance.

## CUDA or MPS is slower than CPU

Accelerators require sufficiently large arrays and long runs to amortize kernel
launches. Increase the production workload before drawing a device conclusion;
do not use a tiny smoke grid as a throughput forecast.

## `torch.compile` starts slowly

The first call compiles the static step graph. The RTX 3060 reference measured
45.1–56.0 seconds for the cold chunk and 1.66–2.15 seconds for the first
remainder graph. Use eager mode for short runs and benchmark compilation only
after excluding warm-up from measured steps.

## Cartopy downloads data

The optional `--coastlines` flag can download Natural Earth data on first use.
Omit the flag in offline environments.

## PyVista cannot open a window

Interactive rendering requires a working display/OpenGL context. Use a static
Matplotlib output, configure off-screen rendering, or run on a desktop session.
The opt-in render test is enabled with

```bash
IONOSPHERE_TEST_PYVISTA_RENDER=1 \
  uv run --extra test --extra visualization --extra verification pytest -q
```

## Diagnose a run

Record the following before reporting a problem:

```python
print(simulation.diagnostics())
```

Also include the command, Git revision, Python/NumPy/PyTorch versions, runtime,
device, dtype, subdivision, radial grid, and whether compilation was enabled.

# Ionosphere geodesic FDTD

`ionosphere-fdtd` is a PyTorch implementation of a three-dimensional
geodesic finite-difference time-domain model for the Earth–ionosphere
waveguide. It combines an icosahedral primal/dual surface mesh with staggered
radial Yee planes, conductive materials, localized current sources, receiver
sampling, and two- and three-dimensional visualization. NumPy remains a
required host-side dependency for mesh construction, portable artifacts, and
scientific analysis; it is not a compute runtime.

![FDTD pulse propagation from Gwangju on the geodesic Earth-ionosphere grid](docs/verification/images/taflove-fig-3-11-gwangju.gif)

## Highlights

- PyTorch CPU, NVIDIA CUDA, and Apple Metal/MPS execution
- Full-spherical radial curls and conservative CFL selection
- Exponential conductive integration with a legacy trapezoidal option
- Static conforming surface and 2:1-balanced radial refinement
- Configurable materials, sources, anomalies, surface impedance, and plasma ADEs
- Mesh-bound material artifacts with explicit dataset provenance
- Two-rank PyTorch/NCCL execution for models that require both local GPUs
- Portable, versioned checkpoints with device-portable restart
- Surface maps, radial sections, receiver traces, interactive 3-D views, and animations
- Analytic Maxwell checks and Simpson–Taflove qualitative verification workflows

## Quick start

Python 3.11 or newer is required.

```bash
uv sync
uv run ionosphere --steps 200
```

PyTorch is a required dependency and the default is CPU with `float64`
fields. This first run requires no accelerator or optional plotting packages.
It prints
the selected mesh, stable time step, field memory, and finite field maxima.

For repeatable runs, copy the CPU-safe starter configuration and edit it instead
of repeating command-line options:

```bash
cp configs/ionosphere.example.toml run.toml
uv run ionosphere --config run.toml
```

This starter remains on PyTorch CPU and writes
`artifacts/runs/demo.npz`. Use
[`configs/ionosphere.research.toml`](configs/ionosphere.research.toml) only
after selecting and sizing the intended accelerator workload.

Render the exact saved state from the starter run:

```bash
uv run --extra visualization ionosphere-visualize \
  --config run.toml surface
```

The resulting checkpoint and surface map are
`artifacts/runs/demo.npz` and `artifacts/figures/demo-surface.png`. Follow the
[learning path](docs/manual/learning-path.md) before moving from this
data-free demonstration to a physical or observational model.

Explicit command-line options override values from the file. See the
[command-line reference](docs/manual/command-line-reference.md#toml-configuration-files)
for the simulation and visualization table layout.

Minimal Python usage:

```python
from ionosphere_fdtd import GeodesicFDTD, GaussianCurrent, SimulationConfig

simulation = GeodesicFDTD(
    SimulationConfig(subdivision=2, radial_cells=24),
    source=GaussianCurrent(carrier_frequency_hz=20.0),
)
simulation.step(1000)
print(simulation.diagnostics())
```

## Documentation

- [User manual](docs/manual/index.md)
- [Installation](docs/manual/installation.md)
- [Quick start](docs/manual/quickstart.md)
- [Learning path](docs/manual/learning-path.md)
- [Command-line reference](docs/manual/command-line-reference.md)
- [Simulation configuration](docs/manual/simulation.md)
- [Materials and sources](docs/manual/materials-and-sources.md)
- [Runtime and performance](docs/manual/backends.md)
- [Visualization and output](docs/manual/visualization-and-output.md)
- [Troubleshooting](docs/manual/troubleshooting.md)
- [PyTorch-only migration and 0.2.0 release notes](docs/manual/pytorch-only-migration.md)
- Verification: [analytic solutions](docs/verification/analytic-solution-benchmarks.md),
  [Simpson–Taflove 2004](docs/verification/simpson-taflove-2004.md), and
  [Simpson–Taflove 2006](docs/verification/simpson-taflove-2006.md)
- [Runtime benchmarks](docs/benchmarks/pytorch-runtime-matrix.md)
- [Refinement strategy](docs/benchmarks/refinement-strategy.md)
- [Two-GPU scaling](docs/benchmarks/distributed-scaling.md)

The verification reports distinguish qualitative agreement with published
phenomena from an independently converged physical model. The Simpson–Taflove
2006 radial trend is currently provisional: its magnitude is not mesh-converged
and is not accepted as a quantitative oil-detection prediction. Exact published
curves require author inputs that are not available in the papers.

## Development

```bash
uv run --extra test --extra visualization --extra verification pytest -q
```

Verification workflows are kept outside the distributed runtime package and
run from a source checkout. The linked verification reports contain their
commands, acceptance criteria, and current results.

## References

1. D. A. Randall et al., “Climate Modeling with Spherical Geodesic Grids,”
   *Computing in Science & Engineering*, 4(5), 32–41, 2002.
2. J. J. Simpson and A. Taflove, “Three-dimensional FDTD modeling of impulsive
   ELF propagation about the entire Earth-sphere,” *IEEE TAP*, 52(2), 443–451,
   2004.
3. J. J. Simpson, R. P. Heikes, and A. Taflove, “FDTD modeling of a novel ELF
   radar for major oil deposits using a three-dimensional geodesic grid of the
   Earth-ionosphere waveguide,” *IEEE TAP*, 54(6), 1734–1741, 2006.
4. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
   Finite-Difference Time-Domain Method*, 3rd ed., Chapter 3, 2005.

## License

Copyright 2026 Kyungwon Chun.

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the
project copyright notice.

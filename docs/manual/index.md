# Ionosphere FDTD User Manual

This manual explains how to install, configure, run, and inspect the
`ionosphere-fdtd` solver. It documents supported user-facing behavior; the
scientific evidence for correctness remains in `docs/verification/`.

## Scientific status

The installed default is a data-free demonstration model, not an observational
Earth product. Verification reports distinguish three different claims:

- qualitative agreement with published phenomena and trends;
- convergence and physical defensibility of an independently specified model;
- exact numerical reproduction of a publication's curves.

The first claim has supporting evidence for the Simpson–Taflove studies. The
Simpson–Taflove 2006 radial oil response is not yet mesh-converged, so its
magnitude is provisional and is not a quantitative detection prediction. Exact
curve reproduction is information-limited by unpublished mesh, material,
source, and receiver details. See the
[2006 verification report](../verification/simpson-taflove-2006.md) for the
current evidence and stopping boundary.

## Contents

1. [Installation](installation.md)
2. [Quick start](quickstart.md)
3. [Learning path](learning-path.md)
4. [Command-line reference](command-line-reference.md)
5. [Simulation configuration](simulation.md)
6. [Materials and sources](materials-and-sources.md)
7. [Runtime and performance](backends.md)
8. [Visualization and output](visualization-and-output.md)
9. [Troubleshooting](troubleshooting.md)
10. [PyTorch-only migration and 0.2.0 release notes](pytorch-only-migration.md)

If this is your first run, stop after the Quick Start has produced both
`demo.npz` and `demo-surface.png`. The Learning Path then explains which model
choice to change next and which checks are required before interpreting a run
quantitatively.

## Interfaces

The project provides two installed commands:

| Command | Purpose |
|---|---|
| `ionosphere` | Run a simulation and print scalar diagnostics |
| `ionosphere-visualize` | Render maps, sections, meshes, animations, and receiver traces |

The Python API exposes the same solver, mesh, material, source, runtime, and
visualization objects through `ionosphere_fdtd`.

The installed commands intentionally expose the compact data-free workflow.
Static adaptive meshes, balanced radial refinement, gridded and mesh-native
materials, surface impedance, magnetized plasma, and generic two-rank execution
are Python APIs. Paper-specific adaptive and distributed runners remain under
the source-only `verification` package.

## Units and field layout

All public values use SI units. The evolving fields are PyTorch tensors:

| Field | Shape association | Time |
|---|---|---|
| `er` | dual cell × radial node | integer electric step |
| `ht` | surface edge × radial node | magnetic half-step |
| `et` | surface edge × radial half-node | integer electric step |
| `hr` | primal triangle × radial half-node | magnetic half-step |

Use `simulation.electric_time_s` for electric fields and
`simulation.magnetic_time_s` for magnetic fields.

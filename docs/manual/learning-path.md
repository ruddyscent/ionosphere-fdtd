# Learning Path

The solver has a short operational quick start and a longer scientific learning
curve. Use the stages below in order; each answers a different question.

## 1. Confirm the installation

Run the CPU starter in the [Quick Start](quickstart.md). At this stage, check
only that the mesh is created, the time step is accepted, field values remain
finite, a checkpoint is written, and the checkpoint can be rendered.

The default model is deliberately small and data-free. Its coarse cells are
appropriate for exercising software, not for resolving a stated physical
wavelength or geographic feature.

## 2. Understand the four fields

The geodesic surface and staggered radial grid store different field
components at different locations and times:

| Name | Physical component | Discrete support |
|---|---|---|
| `er` | radial electric field | surface dual cell and radial node |
| `et` | tangential electric field | surface edge and radial half-node |
| `hr` | radial magnetic field | surface triangle and radial half-node |
| `ht` | tangential magnetic field | surface edge and radial node |

Electric fields are evaluated at integer steps and magnetic fields at
half-steps. A plotted altitude selects the nearest compatible staggered plane;
it does not imply continuous vertical interpolation.

## 3. Choose resolution from the question

Do not begin with a subdivision number copied from a paper or benchmark.
Identify the highest frequency, smallest material feature, source support, and
receiver support that the model must resolve. Then run

```bash
uv run ionosphere --config run.toml --dry-run
```

to validate the runtime/device and see the exact field allocation before committing
to a long run. Refine horizontal and radial resolution independently and retain
at least three levels when estimating convergence. A finer grid is not evidence
of convergence by itself.

## 4. Select the model class

| Goal | Starting interface |
|---|---|
| Software or plotting smoke test | CPU starter TOML |
| Homogeneous or synthetic propagation | installed CLI or basic Python API |
| Geographic layers or gridded observations | Python material API |
| Surface impedance or magnetized plasma | Python ADE APIs |
| Static local surface/radial refinement | Python mesh builders |
| Paper reproduction | source-only `verification` workflows |

The installed CLI intentionally exposes only the compact data-free model. Use
[Materials and Sources](materials-and-sources.md) for observational inputs and
[Simulation Configuration](simulation.md) for geometry, boundary, material
support, and refinement choices.

## 5. Separate demonstration, verification, and prediction

- A demonstration shows that a workflow runs and produces inspectable fields.
- Verification compares a fully stated numerical model with analytic results,
  convergence evidence, or published qualitative phenomena.
- A physical prediction additionally requires traceable observational inputs,
  appropriate constitutive physics, uncertainty analysis, and receiver/source
  operators matched to the intended measurement.

Record the configuration, package or Git revision, dataset provenance,
checkpoint, receiver definition, and post-processing with every quantitative
result. The paper reports under `docs/verification/` state which conclusions
are supported and which remain information-limited.

## 6. Move to an accelerator last

Once the CPU model and outputs are understood, compare PyTorch devices on the
intended grid. Use CUDA or MPS only when the workload offsets framework and
kernel-launch overhead. Compilation is intended for long fixed-shape runs and
should be benchmarked separately from eager execution. See
[Runtime and Performance](backends.md) before using the research template or
the two-GPU runner.

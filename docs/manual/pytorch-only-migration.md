# PyTorch-only Migration and 0.2.0 Release Notes

Version 0.2.0 is a breaking runtime release. PyTorch is now the only compute
runtime for fields, coefficients, sources, observations, optional-physics ADE
state, and distributed execution. NumPy remains required for host-side mesh and
material preprocessing, portable NPZ storage, visualization, and scientific
analysis; it is not a selectable field solver.

## Minimum versions and platforms

| Component | Minimum or support boundary |
|---|---|
| Python | 3.11 |
| NumPy | 2.0; required host-side dependency |
| PyTorch | 2.5; required compute dependency |
| CPU | Supported, `float32` and `float64`; default device is CPU |
| NVIDIA CUDA | CUDA-enabled PyTorch build, compatible driver and GPU; `float32` and `float64` |
| Apple MPS | MPS-enabled PyTorch on supported macOS hardware; `float32` only |
| Distributed | Exactly two local Torch ranks; NCCL is the documented CUDA path |

Install the core CPU runtime from a source checkout with:

```bash
uv sync
uv run ionosphere --subdivision 1 --radial-cells 8 --steps 2
```

The smoke run requires no accelerator. A successful setup line contains
`runtime=torch device=cpu dtype=float64`. Select an appropriate hardware build
of PyTorch before requesting CUDA or MPS.

## Breaking changes

- The public array-backend hierarchy and NumPy compute implementation are
  removed.
- Torch moved from the `pytorch` extra into core dependencies.
- API and CLI defaults are now CPU and `float64`.
- Fields and receiver observations are Torch tensors on the simulation device.
- NumPy export is an explicit, detached terminal operation.
- Checkpoints remain portable, but loading always constructs the current Torch
  runtime.

## Old-to-new reference

| Before 0.2.0 | 0.2.0 replacement |
|---|---|
| `from ionosphere_fdtd import ArrayBackend` | Removed; there is no public runtime abstraction |
| `NumPyBackend`, `TorchBackend`, `create_backend` | Removed; construct `GeodesicFDTD` with `device=` and `dtype=` |
| Imports below `ionosphere_fdtd.backends` | Removed with the package; use public solver APIs |
| Optional install `ionosphere-fdtd[pytorch]` or `.[pytorch]` | Install the project normally; Torch is required |
| `GeodesicFDTD(..., backend="numpy")` | Remove `backend`; the default is Torch CPU/`float64` |
| `GeodesicFDTD(..., backend="torch", device="cuda")` | `GeodesicFDTD(..., device="cuda", dtype="float32")` |
| `GeodesicFDTD.load_checkpoint(..., backend=...)` | Remove `backend`; pass optional `device=` and `dtype=` |
| `checkpoint.load_checkpoint(..., backend=...)` | Same removal and device/dtype replacement |
| `ionosphere --backend ...` or `ionosphere-visualize --backend ...` | Remove the flag; use `--device` and `--dtype` |
| `[ionosphere].backend` or `[visualization].backend` | Remove the TOML key; use `device`, `dtype`, and `torch_compile` |
| CLI/API defaults `device="auto"`, `dtype="auto"` | Defaults are `device="cpu"`, `dtype="float64"`; request auto/`float32` explicitly |
| `simulation.backend.name` | `simulation.runtime` (always `"torch"`) |
| `simulation.backend.device` | `simulation.device` |
| `simulation.backend.dtype_name` | `simulation.dtype_name`; `simulation.dtype` returns the Torch dtype |
| `simulation.backend.threads` | `simulation.threads` |
| `simulation.backend.asarray(...)` and other backend helpers | Use Torch tensor operations or public solver methods; do not import private `_TorchRuntime` |
| `simulation.persistent_backend_bytes` | `simulation.persistent_runtime_bytes` |
| Observation methods returning host NumPy arrays | They return device-native Torch tensors and preserve autograd |
| Implicit host conversion during observation | Call `simulation.to_numpy(values)` only at a terminal export boundary |
| CLI setup output `backend=...` | Setup output reports `runtime=torch device=... dtype=...` |

`BackendUnavailableError` remains a public exception, but it now reports an
unavailable or unsupported PyTorch device/dtype rather than backend selection.
The `backend` parameter accepted by `torch.compile` and the communication
backend accepted by `torch.distributed` are unrelated and remain valid.

Removed CLI flags and TOML keys fail with targeted migration guidance instead
of being ignored. Do not retain a no-op `backend` value in shared
configuration.

## Configuration migration

The portable starter is explicit about its safe defaults:

```toml
[ionosphere]
device = "cpu"
dtype = "float64"
torch_compile = false
torch_compile_chunk_size = 8
```

Choose accelerators deliberately:

```bash
# NVIDIA, after installing a compatible CUDA-enabled Torch build
uv run ionosphere --device cuda:0 --dtype float32 --steps 20000

# Apple MPS; float64 is not supported
uv run ionosphere --device mps --dtype float32 --steps 20000
```

`--device auto` probes CUDA, then MPS, then CPU. It does not consider problem
size, dtype, compile latency, or expected step count and is not a performance
planner.

## Tensor observations and autograd

`record_er_observations()` returns one Torch tensor;
`record_h_observations()` returns a pair of Torch tensors. They remain on the
simulation device and are not synchronized or detached unless requested. Build
the loss before exporting:

```python
traces = simulation.record_er_observations(
    vertex_indices,
    radial_layers,
    weights,
    steps,
    currents=currents,
)
loss = traces.square().mean()
loss.backward()

# Host export is safe only after gradient work is complete.
host_traces = simulation.to_numpy(traces)
```

Supported gradient targets are sampled conductivity/permittivity or direct
update-coefficient tensors, per-step source currents and the Gaussian peak
override, and discrete surface-impedance/plasma coefficient tensors. Mesh
topology, geographic classification, interpolation, file loading, automatic
CFL selection, stored scalar source geometry, and receiver support selection
remain static.

Backward retains intermediate field and ADE states. Memory therefore grows
with the differentiated horizon. Use bounded optimization windows and monitor
peak memory. Detaching fields or saving/reloading a checkpoint between windows
truncates the gradient; do so only when truncated backpropagation is an
intentional approximation.

## Checkpoint compatibility

The v4 writer stores detached host NumPy copies in a pickle-free NPZ archive.
The loader accepts every existing version:

| Version | Behavior |
|---:|---|
| 1 | Restores configuration, material/source metadata, clock, vertices, and fields; mesh topology is reconstructed |
| 2 | Restores exact faces, refinement levels, and topology metadata |
| 3 | Also restores the surface-impedance model and ADE memory |
| 4 | Also restores the mesh-bound plasma model and every species-current ADE state |

Legacy `runtime.backend` metadata is advisory. A v1–v4 archive is loaded onto
the caller-selected Torch `device` and `dtype`; it does not revive NumPy
execution. Numeric fields and ADE state are portable, but autograd graphs and
trainable tensor overrides are never serialized. Recreate trainable tensors
after loading.

## Performance expectations

PyTorch-only does not imply that every workload is faster:

- the historical s2/r16 `float32` result measured Torch CPU at 0.60 times the
  removed NumPy CPU runtime;
- tiny eager CUDA runs can be slower than CPU because launch and synchronization
  overhead dominate;
- accelerator crossover depends on grid, radial cells, dtype, GPU, and run
  length;
- the 2026-08-27 RTX 3060 matrix measured 45.1–56.0 seconds for cold chunk
  compilation and 1.66–2.15 seconds for the first remainder graph;
- on the agreed s4/r40 `float32` baseline, eager and compiled CUDA reached
  1.054 and 1.132 times pre-migration Torch throughput with unchanged
  persistent tensor memory.

Use eager execution for short jobs. Measure compile setup separately and
amortize it only across a sufficiently long, fixed-shape run. Bare-loop
throughput does not predict receiver, checkpoint, surface-impedance, or plasma
cost; use the separate end-to-end workload results in the
[PyTorch runtime matrix](../benchmarks/pytorch-runtime-matrix.md).

Historical JSON and plots containing NumPy rows are preserved unchanged as
pre-migration evidence. Maintained benchmark commands execute only Torch
CPU/CUDA/MPS cases.

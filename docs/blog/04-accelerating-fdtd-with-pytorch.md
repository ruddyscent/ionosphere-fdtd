# FDTD on a Planet, Part 4: Accelerating the Solver with PyTorch

The PyTorch version of this project is not a separate FDTD algorithm. It uses the same mesh, field layout, material coefficients, source placement, Courant check, and four update equations as the NumPy version. Only the small set of array and topology operations changes.

That design gives us two benefits at once: NumPy remains a readable double-precision reference, while PyTorch can run the same time integrator on a multicore CPU, Apple GPU through Metal, or NVIDIA GPU through CUDA. Static field steps can also be compiled with `torch.compile`.

```mermaid
flowchart LR
    S["Shared geodesic FDTD solver"] --> B["PyTorch backend"]
    B --> C["CPU<br/>float32 or float64"]
    B --> M["Apple MPS<br/>float32"]
    B --> G["NVIDIA CUDA<br/>float32 or float64"]
    B --> K["torch.compile<br/>static full graph"]
```

## Backend selection and device policy

PyTorch support is optional:

```bash
uv sync --extra pytorch
```

The backend accepts `cpu`, `mps`, `cuda`, `cuda:N`, and `gpu` as a CUDA alias. With `device="auto"`, it selects CUDA first, then MPS, then CPU.

```python
simulation = GeodesicFDTD(
    config,
    source=source,
    backend="torch",
    device="auto",
)
```

Device resolution is explicit and defensive. Requesting unavailable MPS or CUDA raises a backend error instead of quietly falling back to CPU. An out-of-range CUDA index is also rejected. This prevents a production command from completing slowly on an unintended device while appearing successful.

## Tensor placement is established once

At initialization, all floating-point geometry, coefficients, materials, and fields are converted to the requested dtype and device. Edge and face indices become device-resident `torch.long` tensors. During stepping, no mesh data needs to cross the host–device boundary.

The field update receives only a scalar source current tensor. In compiled mode, the solver precomputes the requested sequence of source-current samples as a backend tensor, then invokes the compiled field graph once per time step. Simulation counters and time are updated outside that graph.

This boundary is intentional: tensor mathematics stays compilable, while Python bookkeeping remains simple.

```mermaid
flowchart TB
    H["Python host<br/>source samples, counters, diagnostics"]
    D["Device-resident static tensors<br/>topology, geometry, materials"]
    F["Device-resident mutable fields<br/>Eᵣ, Eₜ, Hᵣ, Hₜ"]
    U["Eager or compiled field step"]
    H -- "current scalar" --> U
    D --> U
    F <--> U
    F -. "explicit observation only" .-> H
```

## PyTorch topology kernels

Edge and dual-edge differences look almost identical to NumPy:

```python
vertex_values[edges[:, 1]] - vertex_values[edges[:, 0]]
face_values[left_faces] - face_values[right_faces]
```

Triangular face circulation gathers one edge column at a time, multiplies by its sign, and accumulates in place. Because every primal face has exactly three edges, the operation has a fixed structure that works well in a static graph.

Dual-cell circulation uses `index_add_`:

```python
result = torch.zeros(output_shape, dtype=edge_values.dtype, device=device)
result.index_add_(0, edges[:, 0], edge_values)
result.index_add_(0, edges[:, 1], edge_values, alpha=-1.0)
```

Each oriented edge contributes positively to its tail dual cell and negatively to its head dual cell. Unlike the NumPy backend's padded degree-six gather, this is a device scatter reduction. The two forms implement the same topological incidence operator, and the cross-backend tests verify that equivalence through the complete solver.

## Eager execution and compilation

The eager backend is the simplest way to start:

```bash
uv run --extra pytorch ionosphere \
  --backend torch --device mps --steps 200
```

For a long run with static shapes, compilation can fuse and optimize the tensor-only update:

```bash
uv run --extra pytorch ionosphere \
  --backend torch --device cuda --torch-compile --steps 20000
```

The backend calls:

```python
torch.compile(step, fullgraph=True, dynamic=False)
```

`fullgraph=True` requires the complete field update to be captured, which exposes accidental Python graph breaks early. `dynamic=False` tells the compiler that field shapes do not change during the simulation. This is a good match for FDTD: geometry is fixed and the same stencil-like operations repeat thousands of times.

Compilation has a warm-up cost. It should not be assumed faster for the 42- or 162-cell development grids. Its value appears when enough work follows the initial compile or when the grid is large enough to amortize dispatch overhead.

```mermaid
flowchart LR
    P["Prepare fixed-shape tensors"] --> C["Compile first field step"]
    C --> U1["Update 1"] --> U2["Update 2"] --> Un["Update N"]
    C -. "one-time warm-up cost" .-> U1
```

## Precision is device-dependent

The automatic PyTorch dtype is `float32` on every device. Explicit `float64` is supported on CPU and CUDA. Apple MPS does not support double precision in this backend and rejects that combination.

Typical choices are therefore:

```bash
# Fast Apple GPU exploration
uv run --extra pytorch ionosphere \
  --backend torch --device mps --dtype float32 --steps 20000

# Long double-precision CUDA run
uv run --extra pytorch ionosphere \
  --backend torch --device cuda --dtype float64 \
  --torch-compile --steps 35000
```

Single precision halves field storage relative to double precision and usually improves accelerator throughput. It is not always the dominant source of error, however. In the repository's global validation, a matched CUDA `float64` run changed the early attenuation error negligibly relative to MPS `float32`; correcting the ionospheric profile and spectral window mattered far more. Later production verification still used CUDA double precision so that remaining residuals could not be attributed casually to arithmetic precision.

This is the right way to reason about dtype: test it against the observable of interest rather than assuming either “float32 is enough” or “float64 fixes the physics.”

## CPU thread control

PyTorch CPU uses its process-wide intra-op thread count unless `torch_threads` is supplied. The option is valid only for the CPU device:

```bash
uv run --extra pytorch ionosphere \
  --backend torch --device cpu --torch-threads 1 --steps 20000
```

Small arrays can be faster with one thread because thread-pool overhead exceeds useful parallel work. Larger grids may benefit from more threads. There is no universal setting; benchmark the actual subdivision, radial count, dtype, and machine.

The same warning applies when comparing NumPy CPU, PyTorch CPU, MPS, and CUDA. A tiny default problem measures dispatch and framework overhead at least as much as it measures FDTD throughput.

## Keeping observation off the hot path

The public fields are backend-native tensors. Copying a whole CUDA or MPS field to NumPy at every time step would synchronize the device and destroy much of the benefit of asynchronous execution.

The API therefore makes conversion explicit:

```python
er_on_host = simulation.to_numpy(simulation.er)
value = simulation.field_value("er", vertex, layer)
```

Visualization copies tensors to the CPU only when a frame is rendered. Receiver recording can accumulate observations in backend-native storage and synchronize periodically rather than after every scalar sample. Diagnostics synchronize only when values such as a maximum norm are requested.

This separation between simulation cadence and observation cadence is a general GPU lesson: an efficient kernel can still be surrounded by an inefficient measurement loop.

## Cross-backend correctness

Acceleration is useful only if it preserves the algorithm. The test suite constructs matched NumPy and PyTorch CPU simulations in `float64`, advances both for 40 source-driven steps, and compares all four fields:

```python
for field in ("er", "et", "hr", "ht"):
    np.testing.assert_allclose(
        torch_simulation.to_numpy(getattr(torch_simulation, field)),
        getattr(numpy_simulation, field),
        rtol=1e-11,
        atol=1e-12,
    )
```

Additional tests cover eager versus compiled execution, automatic device choice, MPS execution when available, CUDA alias handling, dtype restrictions, face circulation, visualization conversion, and configurable CPU thread count.

The automatic PyTorch `float32` result is also compared with the NumPy `float64` reference through relative $L_2$ field errors. That is a more meaningful test of a precision mode than checking only for finite numbers.

## What production scale looks like

The authoritative archived production run uses subdivision 8: 655,362 surface cells, 40 radial cells, 35,000 steps, compiled PyTorch on an NVIDIA RTX 3060, and `float64`. It completed in 2,677.5 seconds. A refactored compiled kernel avoided the former 10.1 GB compiled-preflight allocation, so that older memory figure should not be treated as the current production requirement.

Those numbers are not universal benchmarks. They document the scale and configuration of one reproducible experiment. Runtime depends on GPU architecture, compiler version, dtype, topology reduction performance, and observation frequency. The more durable conclusion is architectural: a backend-neutral solver allowed the same small tests, physical setup, and validation analysis to move from a NumPy reference to a compiled CUDA production run.

## A complete PyTorch example

```python
from ionosphere_fdtd import GeodesicFDTD, GaussianCurrent, SimulationConfig

simulation = GeodesicFDTD(
    SimulationConfig(subdivision=3, radial_cells=40),
    source=GaussianCurrent(
        latitude_deg=35.1595,
        longitude_deg=126.8526,
        altitude_m=2_500.0,
        carrier_frequency_hz=20.0,
    ),
    backend="torch",
    device="auto",
    dtype="float32",
    compile_step=True,
)

simulation.step(20_000)
simulation.backend.synchronize()
print(simulation.diagnostics())
surface = simulation.to_numpy(simulation.er[:, 20])
```

For a very small grid, start without `compile_step=True`, measure eager execution, and add compilation only when the run is long enough to justify it.

## The broader lesson

GPU acceleration did not require rewriting the geophysical model or maintaining two versions of Maxwell's equations. It required identifying a compact algebra of topology operations, making all persistent state backend-native, keeping transfers explicit, and validating the accelerated path against a simple reference.

That closes the first four-part series:

1. the Earth–ionosphere waveguide defines the physical and engineering problem;
2. the geodesic primal–dual grid turns the sphere into oriented integral operators;
3. NumPy provides a vectorized, inspectable reference implementation;
4. PyTorch carries the same algorithm to CPU, MPS, and CUDA at production scale.

The next natural work is not another backend. It is improving the model and its evidence: finer convergence studies, more realistic ionospheric physics, better crustal data, and continued separation between numerical error and uncertainty in the physical inputs.

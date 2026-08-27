# Runtime and Performance

## Supported combinations

| Runtime | Device | Dtypes | Notes |
|---|---|---|---|
| PyTorch | CPU | `float32`, `float64` | Thread count is configurable |
| PyTorch | CUDA | `float32`, `float64` | `cuda:N` selects a specific GPU |
| PyTorch | Metal/MPS | `float32` | `float64` is unsupported |

`device="auto"` chooses CUDA, then MPS, then CPU. The `gpu` alias means CUDA.
The API and CLI defaults are `device="cpu"` and `dtype="float64"`;
accelerators and `float32` are explicit choices.

## CLI examples

```bash
# PyTorch CPU with the default precision
uv run ionosphere --device cpu --dtype float64 --steps 1000

# PyTorch CPU with one intra-op thread
uv run ionosphere \
  --device cpu --torch-threads 1 --dtype float32 --steps 1000

# NVIDIA CUDA
uv run ionosphere \
  --device cuda --dtype float32 --steps 1000

# Apple Metal
uv run ionosphere \
  --device mps --dtype float32 --steps 1000
```

## Two-GPU execution

The distributed solver assigns complete radial columns to two surface
partitions and exchanges only the electric or magnetic ghost rows required by
the next curl. Launch the paired Simpson 2006 adaptive run with one process per
GPU:

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  -m verification.simpson_taflove_2006.distributed_run \
  --output-dir artifacts/verification/adaptive-s10 \
  --target-subdivision 10 \
  --etopo5-path /path/to/ETOPO5.DAT \
  --dtype float64
```

The dedicated runner uses NCCL and requires exactly two torch ranks. Rank-local
CUDA devices come from
`LOCAL_RANK`; do not rely on `nvidia-smi` ordering. `--capacities A B` changes
the target surface workload ratio when the two GPUs have different measured
throughput or usable memory. Reference and anomaly cases run sequentially on
the same mesh so their signatures and anomaly subtraction remain compatible.
The radar runner captures chunks of 32 complete field steps, including their
NCCL halo exchanges, in a CUDA Graph by default; its default observation
interval is also 32 steps. Use `--cuda-graph-chunk-size 0` to disable capture.
Intervals shorter than the graph chunk, including per-step observations, run
their remainder eagerly rather than requiring a chunk size of 1.
See the [two-GPU benchmark](../benchmarks/distributed-scaling.md) before choosing
distributed execution for a mesh that fits on one GPU.

The documented and benchmarked path uses two GPUs in one host. Multi-node
operation has not been validated as a supported workflow. Magnetized-plasma
current halos are also not implemented, so distributed construction rejects a
plasma model instead of omitting its coupling.

## Compilation

`--torch-compile` captures several static field steps in each PyTorch compiled
graph. The default `--torch-compile-chunk-size 8` turns, for example, 80 time
steps into 10 compiled calls instead of 80. A remainder shorter than the chunk
uses the compiled single-step graph, so every requested count is supported
without compiling a new graph shape.

Larger chunks reduce Python dispatch and accelerator launch overhead but
increase compilation time and graph size. Benchmark powers of two such as 4,
8, 16, and 32 on the target grid and device. Compilation has a significant
first-use cost and is intended for long runs with fixed shapes. Warm up with at
least one complete chunk before timing, and compare eager and compiled
execution as separate experiments.

## Choosing a device

- Use PyTorch CPU for installation checks, small grids, and transparent analysis.
- Benchmark CPU for small and medium grids; framework and launch overhead can
  exceed the arithmetic saved by an accelerator.
- Use CUDA or MPS when the grid and step count are large enough to amortize
  kernel-launch overhead.
- Use `float64` for quantitative verification. Use `float32` when memory and
  throughput matter and the resulting numerical tolerance is acceptable.

The historical pre-migration s2/r16 `float32` measurement put PyTorch CPU at
0.60 times the removed NumPy runtime. That is a documented migration cost, not
a reason to expect every PyTorch workload to improve. On the RTX 3060 live
matrix, eager CUDA becomes useful as the grid grows, while compiled CUDA
delivers the largest steady-state gain on long fixed-shape runs. Cold chunk
compilation took 45.1–56.0 seconds and the first remainder graph took
1.66–2.15 seconds, so a short run should remain eager. See the
[runtime matrix](../benchmarks/pytorch-runtime-matrix.md) for the exact cases,
memory, and historical evidence.

## Forward-only and differentiable execution

Ordinary simulations are forward-only Torch computations. Trainable material,
source-current, surface-impedance, and plasma coefficient tensors retain their
autograd graphs through eager and compiled stepping. Keep receiver traces as
Torch tensors while constructing a loss; `simulation.to_numpy(...)` is a
detaching terminal export. Long horizons retain every state needed by backward
and can exhaust memory, so optimize in bounded windows and detach or checkpoint
only between windows where truncated gradients are scientifically acceptable.
See [Materials and Sources](materials-and-sources.md#differentiable-material-parameters).

## Benchmarking

Run the standardized device matrix:

```bash
uv run python -m benchmarks.runtime_matrix \
  --subdivision 2 --radial-cells 16 \
  --steps 200 --warmup-steps 20 --repeats 3 \
  --dtype float32 --torch-compile --torch-compile-chunk-size 8
```

The benchmark excludes setup and transfers from the timed region and
synchronizes CUDA/MPS before stopping the clock. See
[the runtime matrix](../benchmarks/pytorch-runtime-matrix.md) for the full
method and reference run.

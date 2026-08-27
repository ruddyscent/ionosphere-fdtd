# PyTorch Runtime Verification Matrix

## Scope

The maintained benchmark measures the PyTorch-only runtime across device,
dtype, eager/compiled mode, and workload. Live rows use:

| Runtime | Device | Dtypes | Availability |
|---|---|---|---|
| PyTorch | CPU | `float32`, `float64` | Required |
| PyTorch | NVIDIA CUDA | `float32`, `float64` | CUDA-enabled PyTorch |
| PyTorch | Apple Metal (MPS) | `float32` | Apple silicon/macOS |

The scaling defaults are the agreed s2/r16, s4/r40, s6/r80, and s7/r80
production shapes, both dtypes, and three synchronized repeats. Bare
`GeodesicFDTD.step()` throughput is reported separately from end-to-end
source/receiver, diagnostics/checkpoint, surface-impedance, and plasma
workloads.

## Method

Each comparable row uses identical mesh dimensions, radial cells, initial
pseudo-random fields, dtype, mode, workload, warm-up steps, measured steps,
and repeat count. CUDA and MPS are synchronized before the timer stops. The
primary metric is

$$
\text{throughput}=\frac{N_{\mathrm{steps}}}{t_{\mathrm{median}}}.
$$

The schema records initialization, cold chunk compilation, first remainder
graph compilation, every synchronized repeat, median throughput, persistent
tensor memory, process high-water RSS, and CUDA peak allocation. Results from
different devices, dtypes, modes, workloads, or mesh sizes are not
interchangeable.

## 2026-08-27 PyTorch-only verification run

The checked-in live matrix was measured on Linux with PyTorch 2.13.0+cu130
and an NVIDIA GeForce RTX 3060. Every row used 32 measured steps, 32 warm-up
steps, and three synchronized repeats. Compiled rows used a cold cache and a
32-step chunk.

| Grid | Dtype | CUDA eager | CUDA compiled |
|---|---|---:|---:|
| s2 / r16 | `float32` | 1010.2 | 34630.0 |
| s2 / r16 | `float64` | 1018.5 | 32008.0 |
| s4 / r40 | `float32` | 951.7 | 12892.4 |
| s4 / r40 | `float64` | 551.9 | 2549.0 |
| s6 / r80 | `float32` | 82.0 | 508.0 |
| s6 / r80 | `float64` | 41.6 | 100.7 |
| s7 / r80 | `float32` | 20.7 | 127.5 |
| s7 / r80 | `float64` | 10.5 | 25.1 |

Values are steady-state steps/s. Cold chunk compilation took 45.1--56.0
seconds and first remainder-graph compilation took 1.66--2.15 seconds.
The complete record, including persistent tensor, peak process, and peak CUDA
memory, is
[`pytorch-runtime-matrix-rtx3060.json`](../../artifacts/benchmarks/pytorch-runtime-matrix-rtx3060.json).

The like-for-like pre-migration comparison uses the longer historical method
(256 measured steps and 64 warm-up steps). The agreed s4/r40 `float32`
eager and compiled cases reached 1.054 and 1.132 times their historical
throughput while retaining exactly the same persistent tensor memory, so both
are within the 5% threshold. The machine-readable comparison also preserves
the non-gating s2/r16 and `float64` results, including slower measurements,
rather than hiding variance:
[`pytorch-runtime-baseline-rtx3060.json`](../../artifacts/benchmarks/pytorch-runtime-baseline-rtx3060.json).

The end-to-end s2/r16 `float32` run records CPU and CUDA bare-loop,
source/receiver, diagnostics/checkpoint, surface-impedance, and plasma
throughput separately. All ten available rows completed three synchronized
repeats; all five MPS rows are explicitly unavailable on this Linux host.
The complete timings and memory records are
[`pytorch-runtime-end-to-end-s2-r16-float32.json`](../../artifacts/benchmarks/pytorch-runtime-end-to-end-s2-r16-float32.json).

## Historical pre-migration evidence

The JSON artifacts whose names contain `backend`, NumPy rows below, and
native-fast-path prototype results are immutable historical evidence. They
remain available to audit the migration but are not produced by a live NumPy
runtime and do not describe a supported selector.

### 2026-08-14 reference run

The repository includes a 2026-08-14 eager `float32` reference run on Linux
x86-64 with NumPy 2.5.1 and PyTorch 2.13.0+cu130. The intentionally small
subdivision-2 workload exposes framework and kernel-launch overhead:

| Backend | Device | Steps/s | Relative to NumPy | Status |
|---|---|---:|---:|---|
| NumPy | CPU | 3022.4 | 1.00× | available |
| PyTorch | CPU | 1822.5 | 0.60× | available |
| PyTorch | CUDA | 1193.8 | 0.39× | available |
| PyTorch | Metal/MPS | — | — | unavailable on this host |

For this small grid, accelerator launch overhead exceeds the saved arithmetic;
the table must not be generalized to production subdivisions. The complete
machine-readable record is
[`artifacts/benchmarks/backend-matrix-float32.json`](../../artifacts/benchmarks/backend-matrix-float32.json).

### Compiled chunk sweep on CUDA

A 2026-08-15 follow-up measured the multi-step compiled graph introduced after
the eager reference run. The sweep retained the subdivision-2, 16-radial-cell,
`float32` workload, increased each measured interval to 512 steps, used 256
warm-up steps, and took the median of five repeats. Both counts are divisible
by every measured chunk size. Compilation and initial host-to-device transfers
remain outside the timed interval.

| Mode | Chunk | RTX 3060 steps/s | RTX 2060 SUPER steps/s |
|---|---:|---:|---:|
| NumPy CPU | — | 3119.8 | 3026.0 |
| CUDA eager | — | 1166.9 | 1133.6 |
| CUDA compiled | 1 | 5432.8 | 5387.4 |
| CUDA compiled | 16 | 29546.6 | 29933.5 |
| CUDA compiled | 32 | 36550.9 | 36877.7 |
| CUDA compiled | 64 | 34551.0 | 40768.2 |
| CUDA compiled | 128 | 43588.3 | 44715.2 |
| CUDA compiled | 256 | 45534.1 | 46663.8 |

The tested GPUs used PyTorch 2.13.0+cu130: an RTX 3060 (compute capability
8.6, 12 GB) and RTX 2060 SUPER (compute capability 7.5, 8 GB). Chunk 16 already
delivered 25.3–26.4 times the eager CUDA throughput and 9.5–9.9 times the NumPy
throughput. Chunk 256 produced the highest steady-state result, but only
improved 4–5% over chunk 128 while its first compilation took many minutes on
this host. Chunk 32 or 64 is therefore a more practical latency/throughput
choice for short and medium runs; chunk 128 or 256 is justified only when a
long run can amortize compilation.

The machine-readable records are
[`gpu-chunk-sweep-rtx3060.json`](../../artifacts/benchmarks/gpu-chunk-sweep-rtx3060.json)
and
[`gpu-chunk-sweep-rtx2060-super.json`](../../artifacts/benchmarks/gpu-chunk-sweep-rtx2060-super.json).
This benchmark has no source or observation consumer, so it isolates field-step
batching; it does not measure source-current transfer or observation sampling.

### Native fast-path prototype decision

The 2026-08-22 experiment built a PyTorch-native stepper prototype with
preweighted incidence tables and four reusable full-field workspaces. It also
evaluated single-device CUDA Graph replay independently from TorchInductor.
Every cell below is the median of three synchronized repeats from the same
source tree. Chunked modes use 32 steps. The subdivision-2 `float32` and
subdivision-6 rows ran on an RTX 2060 SUPER; subdivision-2 `float64` and
subdivision-4 ran on an RTX 3060. Modes within each row always use the same
GPU, initial fields, dtype, warm-up, and measured step count.

| Grid | Dtype | NumPy CPU | CUDA eager | Compiled | Native eager | Native compiled | CUDA Graph |
|---|---|---:|---:|---:|---:|---:|---:|
| s2 / r16 | `float32` | 3058.7 | 1175.4 | 36785.5 | 1217.1 | 28514.5 | 6343.3 |
| s2 / r16 | `float64` | 2385.1 | 1147.4 | 28659.1 | 1041.9 | 19786.5 | 9262.2 |
| s4 / r40 | `float32` | 95.5 | 904.6 | 11807.3 | 405.3 | 9398.8 | 2940.3 |
| s4 / r40 | `float64` | 44.7 | 1067.7 | 3055.1 | 1173.8 | 2891.2 | 1339.5 |
| s6 / r80 | `float32` | 3.1 | 91.3 | 544.0 | 96.5 | 420.0 | 96.5 |
| s6 / r80 | `float64` | 1.7 | 48.2 | 148.3 | 51.3 | 133.7 | 51.3 |

Values are steps/s. Native eager improved the production s6 case by 5.7% in
`float32` and 6.3% in `float64`, and improved s4 `float64` by 9.9%. It regressed
s4 `float32` and both native compiled variants were slower than the existing
compiled graph. CUDA Graph replay reduced dispatch cost on small grids but
converged to native-eager throughput on the memory-bound s6 grid. The prototype
was therefore removed rather than retained as a second solver implementation.
The existing generic compiled path remains the recommended CUDA mode for long
runs.

Cold compilation took 50.4--65.0 seconds. CUDA Graph capture took
0.109--1.317 seconds, depending on grid and dtype. The native workspaces equal
the four field arrays in size. On s6/r80, persistent `float32` storage rose
from 225.2 MiB to 340.2 MiB and peak device allocation rose from 417.7 MiB for
generic eager to 458.2 MiB for native eager. These costs are reported directly
by the prototype benchmark schema rather than hidden in framework totals.
Complete measurements are retained as negative-result evidence in
[`torch-fast-path-2026-08-22.json`](../../artifacts/benchmarks/torch-fast-path-2026-08-22.json).

### Temporary-allocation audit

The PyTorch profiler recorded eight eager s4/r40 `float32` steps after four
warm-up steps on the RTX 2060 SUPER. Positive self-allocation bytes fell from
189,505,536 to 178,692,096 (5.7%), while allocation-producing operator calls
rose from 224 to 248 because indexed gathers still dominate. Persistent solver
storage rose from 7,794,904 to 11,645,520 bytes and peak live device allocation
rose from 13,974,016 to 15,336,960 bytes.

The prototype replaced the radial derivative and four curl/update outputs with
single-writer workspaces and preweighted incidence tables. Although those
ownership rules were safe, the profiler shows that indexed topology gathers
remain the primary allocation and memory-traffic target. The small allocation
reduction did not repay the extra persistent memory, divergent update logic,
or compiled regression. This result does not justify retaining the prototype
or adding a custom kernel without a separate deterministic gather and layout
study.

The generic inventory and historical prototype inventory are
[`torch-allocations-generic-s4-r40-float32.json`](../../artifacts/benchmarks/torch-allocations-generic-s4-r40-float32.json)
and
[`torch-allocations-native-s4-r40-float32.json`](../../artifacts/benchmarks/torch-allocations-native-s4-r40-float32.json).
Separate synthetic-input inventories cover the generic
[`surface-impedance`](../../artifacts/benchmarks/torch-allocations-surface-impedance-s2-r16-float32.json)
and
[`plasma`](../../artifacts/benchmarks/torch-allocations-plasma-s2-r16-float32.json)
paths. The maintained runtime continues to use the shared generic update path
for every supported physics configuration.

Reproduce the maintained generic inventory with:

```bash
uv run python -m benchmarks.torch_allocations \
  --device cuda --subdivision 4 --radial-cells 40 \
  --dtype float32 --steps 8 --warmup-steps 4 \
  --output allocations.json
```

Pass `--physics surface-impedance` or `--physics plasma` to audit those paths
with deterministic synthetic inputs. The historical native inventory is
retained for the decision record but is not a maintained execution mode.

## Live reproduction

Run every end-to-end workload on one grid in eager mode:

```bash
python -m benchmarks.runtime_matrix \
  --subdivision 2 \
  --radial-cells 16 \
  --steps 32 \
  --warmup-steps 32 \
  --repeats 3 \
  --dtype float32 \
  --workloads bare,source-observation,diagnostics-checkpoint,surface-impedance,plasma \
  --output artifacts/benchmarks/pytorch-runtime-end-to-end.json
```

Measure compiled chunk and remainder graphs separately:

```bash
python -m benchmarks.runtime_matrix \
  --subdivision 2 \
  --radial-cells 16 \
  --steps 32 \
  --warmup-steps 32 \
  --repeats 3 \
  --dtype float32 \
  --torch-compile \
  --torch-compile-chunk-size 32 \
  --workloads bare \
  --output artifacts/benchmarks/pytorch-runtime-compiled.json
```

Unavailable devices are recorded as `unavailable`, not silently omitted. Run
the same command on an Apple-silicon host to populate the MPS row and on an
NVIDIA host to populate CUDA. Hardware-specific results should remain JSON
artifacts; this document defines the stable comparison method rather than
presenting one machine's numbers as universal performance.

### Production-size scaling sweep

Use the isolated-worker scaling benchmark for crossover analysis across mesh
and radial sizes:

```bash
uv run python -m benchmarks.runtime_scaling \
  --grids 2:16,4:40,6:80,7:80 \
  --dtypes float32,float64 \
  --implementations torch-cpu,cuda,mps \
  --modes eager,compiled \
  --workloads bare \
  --steps 32 \
  --warmup-steps 32 \
  --repeats 3 \
  --torch-compile-chunk-size 32 \
  --baseline artifacts/benchmarks/torch-fast-path-2026-08-22.json \
  --output artifacts/benchmarks/pytorch-runtime-scaling.json
```

Each case runs in a fresh process. This makes process peak resident memory
comparable between cases and prevents an out-of-memory exit or timeout from
discarding the rest of the sweep. Results are written atomically after every
case and `--resume` is enabled by default. Compiled cases use a fresh
TorchInductor cache by default, so `cold_compile_seconds` measures a cold
first chunk and `remainder_compile_seconds` measures the first single-step
remainder graph; pass `--no-cold-compile` to study cache reuse instead.

The scaling schema separates:

- `initialization_seconds`: solver construction, initial field upload, and
  device synchronization;
- `cold_compile_seconds`: the first synchronized compiled chunk, including
  its execution, or `null` for eager cases;
- `remainder_compile_seconds`: the first synchronized single-step graph,
  including its execution, or `null` for eager cases;
- `repeat_seconds`: all synchronized repeats used to select the median;
- `steps_per_second`: median synchronized steady-state throughput after
  compilation and warm-up;
- `peak_process_memory_bytes`: the worker process high-water resident set;
- `peak_device_memory_bytes`: peak live CUDA tensor allocation when available;
- `persistent_memory_bytes`: fields, coefficients, geometry, and topology
  retained by the solver.

MPS does not support this solver's `float64` mode and is recorded as
`unavailable`. A worker that exceeds `--timeout-seconds` is recorded as
`timeout`, allowing large subdivision-7 cases to fail explicitly rather than
stalling or truncating the complete matrix.

## Historical production-size results

The 2026-08-15 Linux run used an RTX 3060 12 GB, PyTorch 2.13.0+cu130,
NumPy 2.5.1, 32 measured steps, 32 warm-up steps, and three repeats. All 108
available eager cases and all 36 cold-compiled CUDA cases completed. The 36 MPS
cases were recorded as unavailable because this host is not macOS.

![Backend throughput curves](images/backend-scaling-throughput.png)

![Initialization and cold-compile curves](images/backend-scaling-setup-time.png)

![Persistent memory curves](images/backend-scaling-persistent-memory.png)

Representative endpoints show the change in scale:

| Grid | Dtype | NumPy | Torch CPU | CUDA eager | CUDA compiled |
|---|---|---:|---:|---:|---:|
| subdivision 2, radial 16 | float32 | 3167.8 | 1833.9 | 1151.1 | 32614.7 |
| subdivision 4, radial 40 | float32 | 107.7 | 383.6 | 1166.5 | 11806.5 |
| subdivision 6, radial 80 | float32 | 3.3 | 3.8 | 82.1 | 465.1 |
| subdivision 7, radial 80 | float32 | 0.8 | 0.9 | 20.7 | 116.8 |
| subdivision 7, radial 80 | float64 | 0.5 | 0.5 | 10.5 | 31.1 |

Values are steady-state steps/s. Eager CUDA first becomes the fastest tested
backend at subdivision 3 for most configurations, but the exact crossover
depends on radial cells and dtype. For example, subdivision 2 with 80 radial
cells favors Torch CPU in `float32` and CUDA in `float64`.

Cold compilation took 49–60 seconds across the matrix. Compared with the best
eager backend, the estimated break-even run length ranges from about 174,000
steps for subdivision 2/radial 16/`float32` down to about 950 steps for
subdivision 7/radial 80/`float64`. Consequently, changing `device="auto"` from
GPU-first to a size-only rule is not justified: an optimal choice also needs
the requested dtype, compilation mode, expected step count, GPU model, and
whether a compiled graph is already cached. Keep automatic selection unchanged
until those inputs can be represented explicitly at the run-planning layer.

Complete records are
[`backend-scaling-eager-rtx3060.json`](../../artifacts/benchmarks/backend-scaling-eager-rtx3060.json)
and
[`backend-scaling-compiled-rtx3060.json`](../../artifacts/benchmarks/backend-scaling-compiled-rtx3060.json).

## Interpretation

- Compare eager and compiled PyTorch as separate experiments because compilation
  warm-up and graph caches change the cost model.
- Keep warm-up and measured step counts divisible by the compile chunk size to
  isolate the multi-step graph from the single-step remainder graph.
- Sweep chunk sizes on the target hardware; larger graphs reduce dispatch but
  cost more to compile and need not be faster for every grid.
- Use `float32` for a CPU/CUDA/MPS comparison. Use a separate `float64` run
  for CPU and CUDA; MPS will be unavailable.
- Increase subdivision and step count for accelerator studies so kernel time
  dominates Python and launch overhead.
- Benchmark results measure performance only. Solver correctness remains owned
  by pytest and the three canonical verification reports.

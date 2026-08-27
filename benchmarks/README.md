# Analytic-solution performance benchmarks

These source-checkout-only benchmarks measure representative A0--A4 runtime
workloads. They do not apply scientific acceptance thresholds and are not
collected by the default pytest run.

```bash
python -m benchmarks.analytic_solutions --repeats 3
python -m benchmarks.analytic_solutions --repeats 5 --output benchmark.json
```

Each repeat is an isolated workflow and includes setup as well as the reported
step count. A0, A1, A2, and A4 exercise production `GeodesicFDTD` paths. A3
times the periodic lossy auxiliary reference and is labeled accordingly; it
must not be compared as production spherical-solver throughput.

## PyTorch runtime performance tools

`runtime_matrix` compares one grid across CPU/CUDA/MPS, dtype, mode, and
bare versus end-to-end workloads. `runtime_scaling` isolates each agreed
grid, dtype, device, execution mode, and workload in a worker process.
Compilation reports cold chunk and first remainder graphs separately from
steady-state timing, and every synchronized repeat is retained in JSON.

The default scaling grids are s2/r16, s4/r40, s6/r80, and s7/r80. Pass the
historical `torch-fast-path-2026-08-22.json` artifact as `--baseline` to
enforce like-for-like throughput and persistent-memory comparisons. Historical
NumPy rows remain evidence only; neither tool requires a live NumPy runtime.

`torch_allocations` records an eager PyTorch memory profile and emits the
positive self-allocation operators, call counts, input shapes, persistent
memory, and peak device allocation. It can audit the vacuum,
surface-impedance, and synthetic-plasma paths.

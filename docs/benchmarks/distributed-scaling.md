# Two-GPU Distributed Scaling

## Method

The distributed benchmark runs one process per GPU and assigns complete radial
columns to two connected surface partitions. Each step exchanges packed field
halos with NCCL. Independent interior curls execute while those point-to-point
transfers are active; boundary curls wait for the corresponding halo.

The timer excludes mesh construction, partitioning, graph capture, and warm-up.
Each reported duration is the slower rank's synchronized time, and throughput
uses the median of three repeats. Field memory includes the four rank-local
owned-and-ghost field arrays, but excludes coefficients, communication buffers,
and CUDA/NCCL workspace.

## Reference run

The 2026-08-20 reference uses PyTorch 2.13.0+cu130, an RTX 3060 and an RTX 2060
SUPER, subdivision 6, 40 radial cells, `float32`, 20 warm-up steps, and three
100-step intervals. Equal partition capacities were used.

| Two-GPU mode | Graph chunk | Steps/s | Relative | Largest rank field memory |
|---|---:|---:|---:|---:|
| Overlapped eager | — | 204.1 | 1.00× | 30.0 MB |
| NCCL CUDA Graph | 1 | 210.6 | 1.03× | 30.0 MB |

The graph removes repeated Python, launch, and NCCL setup overhead, but the s6
case is already dominated by field kernels. On a smaller s4 trial it gave a
larger launch-overhead benefit; that result is not used as a production scaling
claim.

For context, the existing single RTX 3060 scaling records report 153.8 steps/s
for eager CUDA and 901.6 steps/s for the compiled 32-step path on the same s6,
r40, `float32` dimensions. Those records use 32 measured steps rather than 100,
so they are a decision aid rather than a strict paired speedup experiment. The
two-GPU graph result is 1.37 times the single-GPU eager result but only 0.23
times the compiled result. Multi-GPU execution should therefore be selected to
fit a mesh that does not fit one GPU, or after a representative benchmark shows
a throughput advantage. A fitting mesh should remain on the faster compiled
single-GPU path.

Machine-readable results are in
[`distributed-eager-s6-r40-float32.json`](../../artifacts/benchmarks/distributed-eager-s6-r40-float32.json)
and
[`distributed-cuda-graph-s6-r40-float32.json`](../../artifacts/benchmarks/distributed-cuda-graph-s6-r40-float32.json).
The single-GPU context comes from
[`backend-scaling-eager-rtx3060.json`](../../artifacts/benchmarks/backend-scaling-eager-rtx3060.json)
and
[`backend-scaling-compiled-rtx3060.json`](../../artifacts/benchmarks/backend-scaling-compiled-rtx3060.json).

## Reproduction

Run the eager case:

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  -m benchmarks.distributed_scaling \
  --subdivision 6 --radial-cells 40 \
  --steps 100 --warmup-steps 20 --repeats 3 \
  --dtype float32 \
  --output artifacts/benchmarks/distributed-eager-s6-r40-float32.json
```

Add `--cuda-graph-chunk-size 1` and select a different output path for the graph
case. Increase the chunk only when most calls can advance that many steps;
observation intervals shorter than the graph chunk execute their remainder
eagerly. The dedicated radar runner therefore pairs its default graph chunk of
32 with a default sample interval of 32. Test `--capacities A B` on the target
adaptive mesh because unequal theoretical GPU capability did not improve the
reference mesh: the changed cut and boundary workload outweighed the intended
compute balance.

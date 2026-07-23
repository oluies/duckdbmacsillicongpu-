# Phase 0 spike — throwaway, and GATING

This directory is the validation spike from the project constitution (Principle I). It exists to
answer one question cheaply: **can the Apple GPU beat CPU DuckDB at a heavy relational operator
on the same unified-memory machine, by ≥3× end to end?** If yes, Phase 1 is unlocked. If no, the
project stops here and reports the negative result.

## Verdict: FAIL — project stops at Phase 0

Run on **Apple M5 Pro, 48 GB unified memory**, 2026-07-23, MLX 0.32.0 / DuckDB 1.5.5.

| | |
|---|---|
| Operator | `GROUP BY key → SUM(val_i), SUM(val_d), COUNT(*)` |
| Rows | 300,000,000 (seed 42, 100k distinct keys + 1 NULL-key group) |
| **CPU DuckDB** | **0.83 s** |
| **Apple GPU (MLX, end to end)** | **9.84 s** (Arrow conversion included) |
| Ratio | **0.08× — the GPU is ~12× slower** |
| Correctness | ✅ exact (int SUM + COUNT bit-exact; float SUM worst rel err 3.65e-6 ≤ 1e-3) |
| Gate (≥3×) | ❌ not met |

The GPU path is **numerically correct** but nowhere near fast enough. Per the constitution this
is a documented stop, not a reason to keep building.

## Why it fails (the unified-memory thesis, confirmed)

Breakdown of the 9.84 s GPU run:

| Stage | Time |
|---|---|
| DuckDB → Arrow export | 1.80 s |
| Arrow → NumPy (null-fill + cast) | 1.11 s |
| NumPy → MLX | 0.50 s |
| **— just moving data to the GPU —** | **≈ 3.4 s** |
| argsort (300M) | 1.81 s |
| gather sorted columns | 2.08 s |
| segment boundaries | 0.86 s |
| int64 cumsum (exact SUM) | 0.44 s |
| float32 scatter (double SUM) | 0.28 s |
| **— GPU compute —** | **≈ 5.5 s** |

The decisive fact: **just handing the GPU its data costs ~3.4 s — about 4× the entire 0.83 s it
takes CPU DuckDB to do the whole job.** There is no host-to-device transfer to amortize, because
on Apple Silicon the CPU engine already sits on the data in the same memory. Even a hypothetical
*zero-cost* Metal hash kernel would still lose: 3.4 s conversion vs 0.83 s CPU ≈ 0.24×, and the
Arrow export alone (1.8 s) already exceeds the whole CPU aggregation. To clear the 3× gate the
entire GPU path would have to finish in < 0.28 s, which the data movement alone rules out.

## Data-size sensitivity — bigger does NOT help (measured)

A natural hypothesis is "the dataset just needs to be very large for the GPU to win." The
measurement refutes it for this operator — larger data makes the GPU **worse**, monotonically:

| rows | CPU DuckDB | Apple GPU (end to end) | ratio |
|---:|---:|---:|---:|
| 2 M | 0.025 s | 0.042 s | 0.59× |
| 10 M | 0.051 s | 0.112 s | 0.46× |
| 50 M | 0.153 s | 0.576 s | 0.27× |
| 100 M | 0.275 s | 1.261 s | 0.22× |
| 300 M | 0.832 s | 9.836 s | 0.08× |

CPU DuckDB scales linearly (3× the rows → ~3× the time). The GPU path scales **worse** than
linearly: the Arrow export and null-fill/cast are O(rows), and the sort is O(rows·log rows), so
there is no fixed overhead for large data to amortize away — the dominant costs grow *with* the
data. Scaling up only widens the gap. The 2 M case looks closest to parity only because kernel-
launch/setup overhead is a larger share there; it is still a loss.

## When would the Apple GPU actually win? Arithmetic intensity, not size

The deciding factor is **FLOPs per byte moved**, not row count. A relational
`GROUP BY … SUM, COUNT` is *memory-bound*: it does roughly one add per row and touches each byte
about once, so runtime is set by how fast you can move the columns — and CPU DuckDB is already
sitting on those bytes in unified memory. Moving them into GPU form (~3.4 s) costs more than the
whole CPU job (0.83 s) before any compute. No amount of extra rows changes that ratio.

The GPU pulls ahead only when there is a lot of *compute per byte* — a **compute-bound**,
high-arithmetic-intensity workload where the fixed Arrow/handoff cost is amortized many times
over:

- **Dense linear algebra** — matrix multiply is O(n³) work on O(n²) data; each element loaded is
  reused ~n times.
- **PCA / SVD / covariance**, and similar factorizations built on those matmuls.
- **ML training/inference, FFTs, iterative solvers** — repeated passes of heavy arithmetic over
  data that is loaded once.

These are exactly what MLX/Metal are built for, and exactly what DuckDB's relational engine is
*not*. A useful reframing of the negative result: the win for GPU on Apple Silicon is in the
**analytical/numerical layer on top of DuckDB** (feed a resident column set once, then do many
FLOPs of matrix algebra), not in replacing DuckDB's memory-bound relational operators. If the
goal shifts toward that, it is a different project than the one this spike gated — and worth its
own spike, measured the same honest way.

## Scope of this result

- **Approach**: MLX sort-based group-by (argsort → gather → segmented reduction), the
  no-shader-code approach the plan chose for the spike. A Phase 1 Metal *hash* aggregation would
  cut the ~5.5 s compute, but not the ~3.4 s Arrow/handoff cost that already loses on its own.
- **Interchange**: DuckDB's Arrow export + null-fill/cast. A future zero-copy path could shave
  the conversion, but DuckDB's Arrow export by itself (1.8 s) is already 2× the CPU job.
- **What would change the answer**: a query whose CPU cost is *much* higher relative to the bytes
  moved (so conversion amortizes), or the GPU operating on data DuckDB has *not* already
  materialized. Plain `GROUP BY … SUM, COUNT` over a resident column is not such a case.

## Environment check (T003)

```
duckdb  1.5.5   pyarrow 25.0.0   mlx 0.32.0   numpy 2.5.1
mlx.core.default_device() -> Device(gpu, 0)
```

MLX 0.32 on this Metal backend: **no int64 scatter, no boolean-mask indexing, float64 silently
downcasts to float32** (Apple GPUs have no doubles). These shaped the kernel — int64 SUM via
cumsum-difference (exact), double SUM via per-group float32 scatter (tolerance-compared).

## Reproduce

```sh
source .venv/bin/activate           # from repo root
python spike/generate_data.py       # ~20 s, writes data/bench.duckdb (gitignored)
python spike/bench_groupby.py       # prints the verdict; exit 1 on this FAIL
```

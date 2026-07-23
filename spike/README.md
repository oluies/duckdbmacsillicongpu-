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

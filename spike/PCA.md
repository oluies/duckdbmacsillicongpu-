# Phase 0b spike — the compute-bound regime (PCA / matrix algebra)

The GROUP BY spike ([`README.md`](./README.md)) found the Apple GPU loses badly on a
memory-bound relational aggregation, and predicted the GPU would win instead on
**high-arithmetic-intensity** work — many FLOPs per byte moved. This spike tests that
prediction directly with PCA (the Gram matrix `XᵀX`, O(N·D²), then a tiny eigendecomposition)
over an N×D embedding-style column pulled from DuckDB.

Both paths start from the same DuckDB `FLOAT[D]` column and both pay the Arrow extraction, so
it is a fair "is it worth shipping the resident columns to the GPU?" test. Run on **Apple M5
Pro, 48 GB**, 2026-07-23, MLX 0.32 / DuckDB 1.5.5, matrix **1,000,000 × 512**.

## Result 1 — single PCA is extraction-bound (GPU wins, but < 3×)

| | CPU (NumPy/BLAS) | Apple GPU (MLX, e2e) | ratio |
|---|---|---|---|
| single Gram + eigh | 0.76 s | 0.58 s | **1.31×** — correct, FAIL vs 3× |

Why only 1.3×: the compute is cheap relative to the load. Breakdown of the GPU path:

| stage | time |
|---|---|
| Arrow → NumPy extraction | 0.558 s |
| NumPy → MLX | 0.180 s |
| **— one-time load —** | **≈ 0.74 s** |
| single Gram `XᵀX` (GPU) | 0.081 s |

The Gram matrix is 0.08 s of GPU work behind a 0.74 s load that *both* engines pay. So a
one-shot numerical UDF barely beats CPU — the same "the CPU already sits on the data" tax the
GROUP BY spike hit. Raising D alone doesn't fix it: the load is O(N·D) and grows too.

## Result 2 — load once, compute many: the GPU clears the gate decisively

The realistic pattern for a numerical gateway is to amortize the one-time load over heavy,
repeated compute on the *resident* matrix (iterative PCA / power iteration, a stack of GNN or
MLP layers, repeated transforms). `--iters K` runs K power-iteration matmuls `Xᵀ(XM)`:

| workload | CPU | GPU | ratio | GPU throughput |
|---|---|---|---|---|
| K=0 (single PCA) | 0.76 s | 0.61 s | 1.26× | — |
| K=5 | 6.49 s | 2.67 s | 2.43× | — |
| **K=20** | 14.30 s | 2.62 s | **5.47×** | **12.3 TFLOP/s** (vs CPU 1.9) |
| K=50 | 32.80 s | 5.73 s | 5.73× | — |

Past K≈20 the load is fully amortized and the ratio plateaus at ~5.5× — the GPU's raw matmul
throughput advantage over CPU BLAS on this machine. **This is the mirror image of the GROUP BY
result**: there, scaling made the GPU worse (0.59× → 0.08×); here, more compute-per-load makes
it better (1.3× → 5.5×), exactly as arithmetic intensity predicts.

## What this means

- **The dividing line is FLOPs per byte, and residency.** GPU offload on Apple Silicon pays off
  for compute-bound numerical work (matrix algebra, PCA/SVD, ML, FFT, distance/similarity at
  scale) **only if the data stays resident on the GPU across many operations**. A single op per
  Arrow load lands at ~1.3× and is not worth the complexity.
- **This is a numerical layer *on top of* DuckDB, not a replacement for its relational
  operators.** It motivates a UDF gateway that (a) exposes numerical kernels, (b) keeps matrices
  resident across calls, and (c) falls back to CPU. Design: [`../docs/udf-gateway.md`](../docs/udf-gateway.md).

## Reproduce

```sh
source .venv/bin/activate
python spike/gen_matrix.py --rows 1000000 --dim 512 --db data/m.duckdb
python spike/bench_pca.py  --db data/m.duckdb --iters 0    # single PCA  -> ~1.3x, FAIL
python spike/bench_pca.py  --db data/m.duckdb --iters 20   # compute-many -> ~5x, PASS
```

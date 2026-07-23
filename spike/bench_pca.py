"""Arithmetic-intensity spike: PCA (Gram + eigendecomposition) CPU vs Apple GPU.

Counterpart to bench_groupby.py. The GROUP BY spike showed the GPU loses on a memory-bound
relational aggregation. This one tests the regime the analysis predicted the GPU *should* win:
a compute-bound, high-FLOP-per-byte workload. PCA over an N x D feature/embedding matrix does
the O(N*D^2) Gram matrix (X^T @ X) — each loaded element is reused ~D times — then a tiny
O(D^3) eigendecomposition.

Both paths start from the same DuckDB table and both pay the Arrow extraction cost, so this is
a fair "is it worth shipping the resident columns to the GPU?" comparison:

- CPU: Arrow -> NumPy, then NumPy/BLAS computes X^T @ X and eigh.
- GPU: Arrow -> MLX, then MLX computes X^T @ X on the GPU (eval'd), eigh on a CPU stream
  (D x D is tiny), result materialized to host. All timed.

Correctness: the Gram matrix and the top principal eigenvalues must agree within a float32
tolerance. Reports both timings, the ratio, and whether the GPU cleared the 3x bar.

The single-PCA case is extraction-bound: the O(N*D^2) Gram compute is far cheaper than the
one-time Arrow load both engines pay, so the GPU wins only ~1.3x. The realistic gateway
pattern is "load once, compute many" — pass `--iters K` to run K power-iteration matmuls
(X^T (X M)) on the resident matrix, amortizing the load over heavy reused compute. The ratio
climbs past the 3x gate as K grows (~5x by K=20 on this machine).

Usage:
    python spike/bench_pca.py [--db PATH] [--runs N] [--gate 3.0] [--iters K]
"""

from __future__ import annotations

import argparse
import sys
import time

import duckdb
import mlx.core as mx
import numpy as np

from gen_matrix import DEFAULT_DB
from timing import best_of, timed

GATE = 3.0
TOP_K = 16  # how many leading eigenvalues to cross-check


def _load_matrix_arrow(con):
    """Export the FLOAT[D] column as Arrow and reshape to an (N, D) float32 array."""
    tbl = con.execute("SELECT v FROM t").to_arrow_table()
    col = tbl.column("v").combine_chunks()
    d = col.type.list_size
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(-1, d)


def cpu_pca(con, iters=0):
    X = _load_matrix_arrow(con).astype(np.float32)   # Arrow -> NumPy
    n = X.shape[0]
    gram = (X.T @ X) / n                              # O(N*D^2) on BLAS
    M = gram
    for _ in range(iters):                           # load-once, compute-many (power iteration)
        M = (X.T @ (X @ M)) / n
        M = M / (np.max(np.abs(M)) + 1e-30)          # renormalize: keep float32 finite
    evals = np.linalg.eigvalsh(gram)                 # O(D^3), tiny
    top = np.sort(evals)[::-1][:TOP_K]
    return np.asarray(gram, dtype=np.float64), np.asarray(top, dtype=np.float64), np.asarray(M)


def gpu_pca(con, iters=0):
    Xnp = _load_matrix_arrow(con)                    # Arrow -> NumPy (host)
    X = mx.array(Xnp)                                # -> MLX (unified memory)
    n = X.shape[0]
    gram = (X.T @ X) / n                             # O(N*D^2) on the Apple GPU
    M = gram
    for _ in range(iters):                           # load-once, compute-many (power iteration)
        M = (X.T @ (X @ M)) / n
        M = M / (mx.max(mx.abs(M)) + 1e-30)          # renormalize: keep float32 finite
    mx.eval(gram, M)                                 # force the heavy compute
    # Eigendecomposition is CPU-only in MLX 0.32; D x D is trivial, run on a CPU stream.
    evals = mx.linalg.eigh(gram, stream=mx.cpu)[0]
    mx.eval(evals)
    g_host = np.array(gram, dtype=np.float64)        # materialize result to host
    top = np.sort(np.array(evals, dtype=np.float64))[::-1][:TOP_K]
    return g_host, top, np.array(M)


def main() -> int:
    ap = argparse.ArgumentParser(description="PCA CPU vs Apple GPU (arithmetic-intensity spike).")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--iters", type=int, default=0,
                    help="power-iteration matmuls on the resident matrix (load-once, compute-many)")
    args = ap.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    n, = con.execute("SELECT count(*) FROM t").fetchone()
    d = con.execute("SELECT len(v) FROM t LIMIT 1").fetchone()[0]
    flops = 2 * n * d * d * (1 + 2 * args.iters)     # Gram + 2 matmuls per iter
    print(f"device        = {mx.default_device()}")
    print(f"matrix        = {n:,} x {d}")
    print(f"iters         = {args.iters} (power-iteration matmuls on resident matrix)")
    print(f"FLOPs         = {flops/1e9:.1f} GFLOP per run")
    print(f"runs          = {args.runs} (best-of, after 1 warmup)")
    print("measuring ...")

    g_cpu, top_cpu, m_cpu = cpu_pca(con, args.iters)
    g_gpu, top_gpu, m_gpu = gpu_pca(con, args.iters)

    cpu_times: list = []
    gpu_times: list = []
    for _ in range(args.runs):
        with timed(cpu_times):
            cpu_pca(con, args.iters)
        with timed(gpu_times):
            gpu_pca(con, args.iters)

    cpu_s = best_of(cpu_times)
    gpu_s = best_of(gpu_times)
    ratio = cpu_s / gpu_s

    gram_rel = float(np.max(np.abs(g_cpu - g_gpu)) / (np.max(np.abs(g_cpu)) + 1e-12))
    eig_rel = float(np.max(np.abs(top_cpu - top_gpu)) / (np.max(np.abs(top_cpu)) + 1e-12))
    # Also verify the power-iteration result M — the repeated-matmul path is the
    # heavy compute being timed and headlined, so it must be cross-checked, not the
    # Gram proxy alone. Both sides renormalize each step, so M is O(1)-scaled.
    m_rel = float(np.max(np.abs(m_cpu - m_gpu)) / (np.max(np.abs(m_cpu)) + 1e-12))
    correct = gram_rel < 1e-3 and eig_rel < 1e-3 and m_rel < 1e-3

    print()
    print(f"operator      = pca_gram_eigh")
    print(f"cpu_seconds   = {cpu_s:.4f}  (NumPy/BLAS, Arrow extraction included)")
    print(f"gpu_seconds   = {gpu_s:.4f}  (MLX, Arrow extraction included)")
    print(f"ratio         = {ratio:.2f}x  ({'speedup' if ratio > 1 else 'slowdown'})")
    print(f"cpu_GFLOP/s   = {flops/1e9/cpu_s:.0f}")
    print(f"gpu_GFLOP/s   = {flops/1e9/gpu_s:.0f}")
    print(f"correct       = {correct}  (Gram rel {gram_rel:.2e}, top-{TOP_K} eig rel {eig_rel:.2e}, "
          f"iter-M rel {m_rel:.2e})")

    if not correct:
        print("verdict       = FAIL (incorrect)")
        con.close()
        return 2
    if ratio >= args.gate:
        print(f"verdict       = PASS  (>= {args.gate:g}x gate)")
        con.close()
        return 0
    print(f"verdict       = FAIL (under gate: ratio {ratio:.2f} < {args.gate:g})")
    con.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 0 head-to-head: CPU DuckDB vs Apple GPU (tasks T007, T008, T010).

Runs GROUP BY key -> SUM(val_i), SUM(val_d), COUNT(*) two ways over the same
data and prints a verdict against the 3x end-to-end gate.

- CPU: DuckDB executes the aggregation. Timed = the query only.
- GPU: columns are exported from DuckDB as Arrow, converted to MLX, and a
  sort-based group-by runs on the Apple GPU. Timed = export + conversion +
  compute + mx.eval() + materialization back to host (contract G1, FR-018).

Verdict is PASS iff the GPU result is correct AND ratio >= gate. Exit code is 0
on PASS, non-zero on any FAIL. See contracts/spike-cli.md.

Usage:
    python spike/bench_groupby.py [--db PATH] [--runs N] [--gate 3.0]
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import mlx.core as mx
import numpy as np

import config
from timing import best_of, timed
from verify import verify

SENTINEL = config.NULL_KEY_SENTINEL


# --------------------------------------------------------------------------- CPU

CPU_SQL = f"""
    SELECT COALESCE(key, {SENTINEL})        AS gkey,
           CAST(SUM(val_i) AS BIGINT)       AS s_i,
           SUM(val_d)                       AS s_d,
           COUNT(*)                         AS c
    FROM t
    GROUP BY key
"""


def cpu_groupby(con) -> dict:
    tbl = con.execute(CPU_SQL).to_arrow_table()
    gk = tbl.column("gkey").to_numpy(zero_copy_only=False)
    si = tbl.column("s_i").to_numpy(zero_copy_only=False)
    sd = tbl.column("s_d").to_numpy(zero_copy_only=False)
    c = tbl.column("c").to_numpy(zero_copy_only=False)
    return {int(k): (int(a), float(b), int(n)) for k, a, b, n in zip(gk, si, sd, c)}


# --------------------------------------------------------------------------- GPU


def _load_columns_to_mlx(con):
    """Export raw columns as Arrow and convert to MLX arrays (timed as GPU cost).

    NULL keys -> SENTINEL (their own group). NULL val_* -> 0 (SUM skips NULLs;
    COUNT(*) still counts the row). This fill is part of the honest conversion
    cost and stays inside the timed region.
    """
    tbl = con.execute("SELECT key, val_i, val_d FROM t").to_arrow_table()
    key_np = tbl.column("key").fill_null(SENTINEL).to_numpy(zero_copy_only=False).astype(np.int32)
    vi_np = tbl.column("val_i").fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    vd_np = tbl.column("val_d").fill_null(0.0).to_numpy(zero_copy_only=False).astype(np.float32)
    return mx.array(key_np), mx.array(vi_np), mx.array(vd_np)


def gpu_groupby(con) -> dict:
    key, vi, vd = _load_columns_to_mlx(con)
    n = key.shape[0]

    # Sort by key so equal keys are contiguous segments.
    order = mx.argsort(key)
    sk = key[order]
    svi = vi[order]
    svd = vd[order]

    # is_start[i] = 1 where a new group begins (first row, or key changed).
    changed = (sk[1:] != sk[:-1]).astype(mx.int32)
    is_start = mx.concatenate([mx.array([1], dtype=mx.int32), changed])
    seg0 = mx.cumsum(is_start) - 1              # 0-based group id per row
    G = int(seg0[-1].item()) + 1

    # start_pos[g] = row index where group g begins. Only the start row of each
    # group carries a nonzero contribution (is_start * index), so scatter-add by
    # seg0 recovers the start index exactly. int32 scatter (int64 scatter is
    # unsupported on the Metal backend).
    idx = mx.arange(n, dtype=mx.int32)
    contrib = is_start * idx
    start_pos = mx.zeros(G, dtype=mx.int32).at[seg0].add(contrib)
    end_pos = mx.concatenate([start_pos[1:] - 1, mx.array([n - 1], dtype=mx.int32)])

    # Integer SUM: exact via int64 cumsum difference (no float, no int64 scatter).
    # sum(group) = cvi_ext[end+1] - cvi_ext[start], where cvi_ext has a 0 prepended.
    cvi = mx.cumsum(svi)
    cvi_ext = mx.concatenate([mx.array([0], dtype=mx.int64), cvi])
    gsum_i = cvi_ext[end_pos + 1] - cvi_ext[start_pos]

    # COUNT: exact from positions.
    gcount = (end_pos + 1 - start_pos).astype(mx.int64)

    # Float SUM: per-group float32 scatter-add. Accumulating within each group
    # keeps magnitudes at group scale (a global float32 cumsum would cancel
    # catastrophically). Order-dependent, hence tolerance-compared.
    gsum_d = mx.zeros(G, dtype=mx.float32).at[seg0].add(svd)

    gkey = sk[start_pos]

    mx.eval(gkey, gsum_i, gsum_d, gcount)  # force lazy graph inside the timed region

    # Materialize to host for comparison (also part of the honest GPU cost).
    hk = np.array(gkey)
    hi = np.array(gsum_i)
    hd = np.array(gsum_d)
    hc = np.array(gcount)
    return {int(k): (int(a), float(b), int(n)) for k, a, b, n in zip(hk, hi, hd, hc)}


# -------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU DuckDB vs Apple GPU group-by gate.")
    ap.add_argument("--db", default=config.DEFAULT_DB)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--gate", type=float, default=config.GATE_RATIO)
    args = ap.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    rows = con.execute("SELECT count(*) FROM t").fetchone()[0]

    print(f"device        = {mx.default_device()}")
    print(f"rows          = {rows:,}")
    print(f"runs          = {args.runs} (best-of, after 1 warmup)")
    print("measuring ...")

    # Warmup (fills caches / JITs kernels; not counted).
    cpu_res = cpu_groupby(con)
    gpu_res = gpu_groupby(con)

    cpu_times: list = []
    gpu_times: list = []
    for _ in range(args.runs):
        with timed(cpu_times):
            cpu_groupby(con)
        with timed(gpu_times):
            gpu_groupby(con)

    cpu_s = best_of(cpu_times)
    gpu_s = best_of(gpu_times)  # includes Arrow conversion, compute, eval, materialize
    ratio = cpu_s / gpu_s

    correct, detail = verify(cpu_res, gpu_res)

    print()
    print(f"operator      = group_by_sum_count")
    print(f"cpu_seconds   = {cpu_s:.4f}")
    print(f"gpu_seconds   = {gpu_s:.4f}  (Arrow conversion included)")
    print(f"ratio         = {ratio:.2f}x  ({'speedup' if ratio > 1 else 'slowdown'})")
    print(f"correct       = {correct}  ({detail})")

    if not correct:
        print("verdict       = FAIL (incorrect)")
        print("A speedup is never reported for a wrong answer. Fix correctness first.")
        con.close()
        return 2

    if ratio >= args.gate:
        print(f"verdict       = PASS  (>= {args.gate:g}x gate)")
        con.close()
        return 0

    print(f"verdict       = FAIL (under gate: ratio {ratio:.2f} < {args.gate:g})")
    print(
        "Per the project constitution, the GPU path did not clear the "
        f"{args.gate:g}x end-to-end bar. The project is expected to STOP at Phase 0."
    )
    con.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Generate a reproducible embedding-style matrix in DuckDB (arithmetic-intensity spike).

One column `v FLOAT[D]` of N rows — the realistic "resident feature/embedding column"
case. This is the input for the PCA / Gram-matrix benchmark (bench_pca.py), which tests the
regime the GROUP BY spike identified as GPU-favorable: high FLOPs per byte moved.

Deterministic from a seed via DuckDB hash(), so CPU and GPU read identical bytes. Not
committed (gitignored).

Usage:
    python spike/gen_matrix.py [--rows N] [--dim D] [--seed S] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import time

import duckdb

DEFAULT_DB = "data/matrix.duckdb"


def build(db_path: str, rows: int, dim: int, seed: int) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute(
        f"""
        CREATE TABLE t AS
        SELECT CAST(
                 list_transform(range({dim}),
                                j -> ((hash(i * {dim} + j, {seed}) % 1000) / 1000.0))
                 AS FLOAT[{dim}]
               ) AS v
        FROM range({rows}) AS r(i);
        """
    )
    n = con.execute("SELECT count(*) FROM t").fetchone()[0]
    con.close()
    gib = rows * dim * 4 / 1024**3
    print(f"db     = {db_path}")
    print(f"rows   = {n:,}")
    print(f"dim    = {dim}")
    print(f"matrix = {n:,} x {dim}  (~{gib:.2f} GiB float32)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the spike embedding matrix.")
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()
    t0 = time.perf_counter()
    build(args.db, args.rows, args.dim, args.seed)
    print(f"generated in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

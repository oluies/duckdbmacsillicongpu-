"""Generate the reproducible benchmark dataset (task T006, spec FR-001).

Writes a DuckDB table `t(key INT, val_i BIGINT, val_d DOUBLE)` of ROWS rows,
deterministically from SEED via DuckDB's own hash(), so the CPU and GPU paths
read identical bytes and the set is regenerable. Not committed (gitignored).

Usage:
    python spike/generate_data.py [--rows N] [--seed S] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import time

import duckdb

import config


def build(db_path: str, rows: int, seed: int, cardinality: int) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    # Decorrelate the columns by hashing i with a distinct salt per column.
    # DuckDB hash(a, b) is a deterministic UBIGINT over both args, so this is
    # reproducible and avoids the int64 overflow that i*bigsalt would hit at
    # a few hundred million rows. The seed enters as a salt so different seeds
    # give different data.
    con.execute(
        f"""
        CREATE TABLE t AS
        SELECT
            CASE WHEN (hash(i, {seed}, 11) % {config.NULL_KEY_EVERY}) = 0
                 THEN NULL
                 ELSE (hash(i, {seed}, 1) % {cardinality})::INTEGER
            END AS key,
            (hash(i, {seed}, 2) % 1000)::BIGINT AS val_i,
            CASE WHEN (hash(i, {seed}, 13) % {config.NULL_VALD_EVERY}) = 0
                 THEN NULL
                 ELSE ((hash(i, {seed}, 3) % 100000) / 100.0)
            END AS val_d
        FROM range({rows}) AS r(i);
        """
    )
    n, distinct_keys, null_keys, null_vald = con.execute(
        """
        SELECT count(*),
               count(DISTINCT key),
               count(*) FILTER (WHERE key IS NULL),
               count(*) FILTER (WHERE val_d IS NULL)
        FROM t
        """
    ).fetchone()
    con.close()

    print(f"db            = {db_path}")
    print(f"rows          = {n:,}")
    print(f"distinct keys = {distinct_keys:,} (+1 NULL-key group)")
    print(f"NULL keys     = {null_keys:,}")
    print(f"NULL val_d    = {null_vald:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the spike benchmark dataset.")
    ap.add_argument("--rows", type=int, default=config.ROWS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--db", default=config.DEFAULT_DB)
    ap.add_argument("--cardinality", type=int, default=config.CARDINALITY)
    args = ap.parse_args()

    t0 = time.perf_counter()
    build(args.db, args.rows, args.seed, args.cardinality)
    print(f"generated in    {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

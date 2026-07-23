# Quickstart: Phase 0 Spike

Get a go/no-go number for the whole project on one Apple Silicon machine.

## Prerequisites

- Stock Apple Silicon Mac (unified memory). No NVIDIA, no CUDA.
- The project venv (already set up at repo root): `duckdb`, `pyarrow`, `mlx`, `numpy`.

```sh
# from repo root; the venv already exists
source .venv/bin/activate
python -c "import duckdb, pyarrow, mlx.core as mx, numpy; print('ok', mx.default_device())"
```

## Run

```sh
# 1. Generate the reproducible dataset (~300M rows; a few GB, regenerable, gitignored)
python spike/generate_data.py            # --rows / --seed / --db to tune

# 2. Head-to-head: CPU DuckDB vs Apple GPU, with the 3x gate
python spike/bench_groupby.py            # exits 0 on PASS, non-zero on FAIL
```

## Reading the result

The benchmark prints a record like:

```text
operator=group_by_sum_count rows=300000000 runs=5
cpu_seconds=<t_cpu>  gpu_seconds=<t_gpu, Arrow conversion included>
ratio=<t_cpu/t_gpu> (speedup)   correct=true
verdict=PASS
```

- **PASS** → the GPU beat CPU DuckDB by ≥3× end to end on a verified-correct result. Phase 1 is
  unlocked.
- **FAIL (under gate)** → correct but under 3×. Per the constitution this is a **documented
  stop**; the project is not expected to proceed.
- **FAIL (incorrect)** → the GPU answer diverged from DuckDB. Fix correctness before any timing
  is meaningful; a speedup is never reported for a wrong answer.

## What this does and does not prove

- It puts a **floor** under GPU performance using a simple sort-based group-by (MLX primitives,
  no custom kernels). A hash-based Metal kernel in Phase 1 can only do better.
- It does **not** build any library or extension. Those are gated on this verdict by design —
  see `plan.md` § Constitution Check and the project constitution.

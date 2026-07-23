# Contract: Phase 0 Spike CLI

The spike's only external interface is a command line and its printed verdict. This contract
is what Phase 0 acceptance is checked against.

## Commands

```text
python spike/generate_data.py [--rows N] [--seed S] [--db PATH]
    Generates the reproducible benchmark dataset into a DuckDB file.
    Defaults: --rows 300_000_000, --seed 42, --db data/bench.duckdb
    Prints: actual row count and distinct-key count.

python spike/bench_groupby.py [--db PATH] [--runs N] [--gate 3.0]
    Runs the CPU-DuckDB vs Apple-GPU head-to-head on GROUP BY key -> SUM, COUNT.
    Defaults: --db data/bench.duckdb, --runs 5 (best-of after 1 warmup), --gate 3.0
    Verifies correctness, then prints the benchmark record and PASS/FAIL verdict.
    Exit code: 0 if verdict is PASS, non-zero if FAIL (correctness or gate).
```

## Output contract (`bench_groupby.py`)

MUST print, in human-readable form, all of:

- `rows` — dataset row count used.
- `cpu_seconds` — DuckDB CPU aggregation wall-clock.
- `gpu_seconds` — GPU end-to-end wall-clock, **Arrow conversion included** (stated as such).
- `ratio` — `cpu_seconds / gpu_seconds`, labeled as speedup (>1) or slowdown (<1).
- `runs` — the N used.
- `correct` — whether the GPU result matched DuckDB (exact int SUM/COUNT, tolerance float SUM).
- `verdict` — `PASS` or `FAIL`, with the reason on FAIL:
  - `FAIL (incorrect)` — GPU result diverged from DuckDB.
  - `FAIL (under gate: ratio X < 3.0)` — correct but not fast enough; MUST also print the
    sentence that the project is expected to **stop at Phase 0**.

## Behavioral guarantees

- **G1 (honest timing)**: `gpu_seconds` MUST span from before Arrow export to after the GPU
  result is materialized to host, including `mx.eval()`. It MUST NOT be kernel-only. (FR-005,
  FR-018)
- **G2 (correctness gates the verdict)**: a `ratio ≥ 3.0` with `correct = false` MUST print
  `FAIL (incorrect)` and exit non-zero. A speedup is never reported for a wrong answer. (FR-004,
  SC-003)
- **G3 (no NVIDIA/CUDA)**: MUST run to completion on a machine with no NVIDIA hardware and no
  CUDA. (FR-006)
- **G4 (reproducible)**: same `--seed` and `--rows` produce byte-identical input for CPU and GPU
  runs. (FR-001)
- **G5 (explicit stop)**: on a correct-but-under-gate result, output states plainly that the bar
  was not met and the project is expected to stop at Phase 0. (FR-007)

## Non-contract

The spike exposes no library API, no importable interface, and no persisted service. It is a
throwaway. If its verdict is FAIL, the scripts are archived with the negative result, not built
upon.

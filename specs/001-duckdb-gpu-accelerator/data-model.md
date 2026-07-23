# Data Model: DuckDB Apple Silicon GPU Accelerator

The "data" here is the benchmark dataset, the aggregation result used both as answer and as
correctness oracle, and the benchmark record that carries the gate verdict. Field types are
concrete because the whole exercise turns on exact numerical behavior.

## Entity: BenchmarkDataset

A reproducible table generated from a fixed seed. Not committed; regenerable.

| Field | Type | Notes |
|---|---|---|
| `key` | INT32 (also an INT64 variant) | Group key. Moderate cardinality (~10⁴–10⁶ distinct). May include NULL to exercise null-group handling. |
| `val_i` | INT64 | Integer value column; exercises exact SUM and overflow behavior. |
| `val_d` | DOUBLE | Floating value column; exercises order-dependent SUM. May include NULL. |

- **Generation rule**: deterministic from a seed constant (fixed in `generate_data.py`); CPU and
  GPU runs read identical bytes.
- **Scale**: ~300M rows (tunable), sized to stay comfortably within unified memory.
- **Validation**: row count and distinct-key count are reported at generation so runs are
  comparable.

## Entity: AggregationResult

The output of `GROUP BY key → SUM(val), COUNT(*)`. Serves as both the returned answer and the
correctness-comparison artifact.

| Field | Type | Notes |
|---|---|---|
| `key` | INT (matches dataset) | Group key, one row per distinct key. NULL key is its own group. |
| `sum_i` | INT64 | Exact integer sum per group. Must equal DuckDB exactly (bit-for-bit). |
| `sum_d` | DOUBLE | Floating sum per group. Compared to DuckDB within a documented tolerance. |
| `count` | INT64 | Row count per group. Must equal DuckDB exactly. |

- **Null semantics**: NULLs in `val_*` are skipped by SUM/COUNT per DuckDB; NULL `key` forms one
  group. Both paths must agree.
- **Ordering**: unordered as produced; both result sets are sorted by `key` before comparison.

## Entity: BenchmarkRecord

One head-to-head measurement.

| Field | Type | Notes |
|---|---|---|
| `operator` | string | e.g. `group_by_sum_count`. |
| `rows` | INT64 | Dataset row count used. |
| `cpu_seconds` | DOUBLE | Wall-clock of the DuckDB CPU aggregation only. |
| `gpu_seconds` | DOUBLE | Wall-clock of Arrow export + conversion + GPU compute + `mx.eval()` + result materialization. |
| `ratio` | DOUBLE | `cpu_seconds / gpu_seconds`. >1 is a GPU win. |
| `runs` | INT | N (best-of-N after warmup); reported alongside the number. |
| `correct` | bool | GPU result matched DuckDB (exact for int SUM/COUNT, tolerance for float SUM). |
| `verdict` | enum `PASS` \| `FAIL` | `PASS` iff `correct` AND `ratio ≥ 3.0`. |

- **Invariant**: `verdict = PASS` requires *both* correctness and `ratio ≥ 3.0`. A fast but wrong
  result is `FAIL`; a correct but sub-3× result is `FAIL` (a documented stop, not rounded up).
- **Invariant**: `gpu_seconds` always includes Arrow conversion; there is no field for
  kernel-only time and none is ever reported as the comparison figure.

## Entity: Operator (Phase 1 forward-looking)

A single GPU relational computation in the eventual library. Present here so Phase 1 tasks have
a shape to fill; not built in Phase 0.

| Field | Type | Notes |
|---|---|---|
| `name` | string | e.g. `group_by_sum_count`, `filter`, `hash_join`. |
| `supported_types` | set | Integer/float types (and later others) the kernel handles exactly. |
| `handles_null` | bool | Whether it matches DuckDB null semantics. |
| `correctness_test` | ref | Test asserting equality vs DuckDB CPU. |
| `benchmark` | ref → BenchmarkRecord | End-to-end timing vs DuckDB CPU. |
| `status` | enum `candidate` \| `retained` \| `removed` | `removed` if its benchmark does not beat CPU. |

- **State transitions**: `candidate → retained` (benchmark PASS) or `candidate → removed`
  (benchmark FAIL). A `removed` operator is deleted, not optimized in place.

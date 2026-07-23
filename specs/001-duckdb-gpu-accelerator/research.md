# Research: DuckDB Apple Silicon GPU Accelerator

Phase 0 research consolidating the decisions the plan depends on. Every "NEEDS CLARIFICATION"
from Technical Context is resolved here.

## R1 — Compute backend for the spike and the library

- **Decision**: MLX for Phase 0; raw Metal / MPS for the Phase 1 library. (Full rationale and
  rejected options are in `plan.md` § Compute Backend Decision — not duplicated here.)
- **Rationale (spike-specific)**: MLX ships `argsort`, `take`/gather, `cumsum`, and segmented
  reductions, which is exactly the primitive set needed to assemble a **sort-based group-by**
  with no shader code, so Phase 0 can be written and measured in hours.
- **Alternatives considered**: MLX-forever (no relational primitives, no strings — prototype
  only); Vulkan/SPIR-V via Cyfra (portable but adds MoltenVK + a JVM boundary the C++ extension
  would cross — rejected until multi-vendor becomes a goal).

## R2 — GPU group-by strategy (sort-based vs hash-based)

- **Decision**: **Sort-based** group-by for the spike: `argsort` the group-key column, gather
  keys+values into sorted order, find segment boundaries where the sorted key changes, then
  segmented-sum and segment-count the values.
- **Rationale**: Every primitive it needs already exists in MLX, so there is no custom kernel
  to write or debug in Phase 0. Hash-based group-by would be faster in theory but needs a GPU
  hash table — that is Phase 1 Metal work, not spike work. The spike's job is a *floor* on GPU
  performance; if even the simple sort-based approach clears 3×, the hash approach only helps.
- **Alternatives considered**: hash aggregation (deferred to Phase 1); DuckDB-internal GPU
  (does not exist; the whole point is to build it).

## R3 — Benchmark dataset

- **Decision**: ~300M rows, fixed seed. Columns: `key` (INT32/INT64 group key, moderate
  cardinality ≈ 10⁴–10⁶ distinct so groups are neither trivially few nor near-unique), and
  `val` (numeric — one integer and one double column so both SUM overflow behavior and
  floating accumulation order can be exercised). Generated with DuckDB's own `range()` +
  hashing so it is reproducible and needs no external data.
- **Rationale**: A few hundred million rows makes the aggregation heavy enough that a real
  CPU/GPU gap is observable above noise, while fitting comfortably in 48 GB unified memory
  (~300M × a few 8-byte columns ≈ single-digit GB). Fixed seed keeps CPU and GPU runs
  comparing identical bytes.
- **Open tuning knobs (not scope)**: exact row count, key cardinality, and column count are
  tunable in `generate_data.py`; they change the *magnitude* of the result, not whether the
  gate is met.

## R4 — DuckDB → GPU interchange

- **Decision**: Apache Arrow via DuckDB's native Arrow export (`.arrow()` / `fetch_arrow_table`
  / `fetch_record_batch`). Convert Arrow buffers to MLX arrays; where an Arrow buffer is a plain
  contiguous primitive array with no nulls, aim for zero-copy into MLX, else one explicit copy.
- **Rationale**: Arrow is DuckDB's supported columnar egress and MLX/NumPy both consume it. The
  conversion is the cost the constitution insists we count — so the benchmark **starts the GPU
  clock before the Arrow export and stops it after the result is back**, capturing conversion
  on both ends.
- **Alternatives considered**: raw DuckDB result fetch into Python lists (far slower, not
  columnar — rejected); DuckDB C API buffer pointers (a Phase 2 zero-copy path, premature now).

## R5 — Timing methodology (avoiding a dishonest GPU number)

- **Decision**: Wall-clock, `perf_counter`. CPU timing wraps only the DuckDB aggregation query.
  GPU timing wraps Arrow export **+** conversion to MLX **+** compute **+** `mx.eval()` (MLX is
  lazy; the result must be forced) **+** materializing the result back to host for comparison.
  Report best-of-N runs after a warmup, and report which N.
- **Rationale**: MLX's laziness is the classic way to accidentally report a 1000× "speedup" that
  is really just an unevaluated graph. Forcing `mx.eval()` inside the timed region and including
  conversion on both ends is what makes the 3× verdict trustworthy (Principle II, FR-018).
- **Alternatives considered**: kernel-only GPU timers (explicitly forbidden by the constitution).

## R6 — Correctness comparison

- **Decision**: Sort both result sets by group key, then assert exact equality on keys and
  counts, and exact equality on integer sums; for floating sums assert bitwise-or-ULP equality
  is *not* required — instead compare against DuckDB's double result with a documented tolerance,
  and separately verify the integer-sum column is exactly equal (integer SUM has a defined exact
  answer; float SUM depends on summation order).
- **Rationale**: Integer SUM and COUNT have exact answers and must match to the bit — this is
  the real correctness signal and catches overflow-handling divergence. Float SUM legitimately
  differs by summation order between CPU and GPU; holding it to a tolerance (and saying so) is
  honest, whereas demanding bitwise float equality would fail for correct code.
- **Alternatives considered**: bitwise equality on all columns (wrong for float SUM); no float
  check at all (loses a real signal — rejected).

## R7 — Reference orchestration design (Sirius upper half)

- **Decision**: Treat Sirius's hardware-agnostic layers (plan interception, CPU fallback,
  pipelining, memory management) as the *reference* for Phase 2's design, adapted to unified
  memory. Do not vendor its CUDA/cuDF compute layer. Phase 0 needs none of this — it is a
  standalone script — so this is documentation for later, not a Phase 0 dependency.
- **Rationale**: Principle III. The novel risk is in compute, not orchestration; reusing a
  proven interception/fallback design keeps original effort on the kernels.

## Resolved unknowns

| Technical Context item | Resolution |
|---|---|
| Phase 1 kernel language | Metal Shading Language (Objective-C++ host), after MLX proves the operator (R1). |
| Group-by algorithm for the spike | Sort-based via MLX primitives (R2). |
| Dataset size / shape | ~300M rows, fixed seed, INT key + INT/DOUBLE values (R3). |
| Interchange + where the clock starts | Arrow export→MLX, GPU clock spans conversion both ends (R4, R5). |
| Float-sum comparison | Exact for integer SUM/COUNT, documented tolerance for float SUM (R6). |

No unresolved NEEDS CLARIFICATION remain.

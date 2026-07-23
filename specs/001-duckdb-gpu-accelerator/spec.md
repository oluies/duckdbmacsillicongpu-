# Feature Specification: DuckDB Apple Silicon GPU Accelerator

**Feature Branch**: `task/task-3-1cc1da`

**Created**: 2026-07-23

**Status**: Draft

**Input**: User description: "Build a system that accelerates DuckDB analytical queries using the GPU on Apple Silicon, falling back to DuckDB's CPU execution for anything unsupported."

## User Scenarios & Testing *(mandatory)*

<!--
  Stories are ordered as strict gates, not just priorities. Phase 0 (US1) must pass
  before Phase 1 (US2) is allowed to start, and Phase 1 must produce a winner before
  Phase 2 (US3) is allowed to start. This gating is a project rule (see Constitution),
  not merely a scheduling preference.
-->

### User Story 1 - Validation spike decides whether the project is worth building (Priority: P1)

A developer evaluating feasibility runs a single script that executes one heavy relational
operator — a `GROUP BY` with `SUM` and `COUNT` — twice over the same large dataset: once in
plain CPU DuckDB, and once on the Apple GPU. The script pulls the columns out of DuckDB as
Arrow, computes the aggregation on the GPU, returns the result, checks it against DuckDB's
own answer, and prints a head-to-head timing with the speedup or slowdown. From that one
number the developer decides whether the full accelerator is worth starting.

**Why this priority**: This is the gating deliverable. Apple Silicon uses unified memory, so
CPU DuckDB already runs in the same memory the GPU would use, with no host-to-device transfer
to amortize. Whether the GPU can beat that baseline at all is unknown, and the entire rest of
the project is contingent on the answer. Building anything larger before this question is
answered risks building on a false premise. If the spike fails the bar, the project is
expected to stop here and report the negative result.

**Independent Test**: Run the single spike script on a stock Apple Silicon machine. It
generates the dataset, runs both paths, verifies numerical equality, and prints both timings,
the ratio, and an explicit pass/fail verdict against the 3× end-to-end bar. Delivers a
go/no-go decision with no other part of the system in place.

**Acceptance Scenarios**:

1. **Given** a stock Apple Silicon machine with no NVIDIA hardware and no CUDA, **When** the
   spike script is run from a clean checkout, **Then** it generates a reproducible dataset of
   a few hundred million rows with integer group keys and numeric values, and proceeds without
   requiring any external hardware.
2. **Given** the generated dataset, **When** the spike runs the `GROUP BY … SUM, COUNT` in
   plain CPU DuckDB, **Then** it records and reports the CPU wall-clock time.
3. **Given** the same dataset, **When** the spike runs the same aggregation on the Apple GPU
   by reading the columns out of DuckDB as Arrow and computing on the GPU, **Then** it records
   and reports the GPU wall-clock time **including** the Arrow conversion cost.
4. **Given** both results, **When** the spike compares them, **Then** it verifies the GPU
   result is numerically identical to the DuckDB result (group keys, sums, and counts all
   match exactly), and fails loudly if they differ.
5. **Given** both timings, **When** the spike reports, **Then** it prints the speedup or
   slowdown and states explicitly whether the GPU path beat CPU DuckDB by at least 3× end to
   end.
6. **Given** a GPU result that does not clear the 3× bar, **When** the spike finishes, **Then**
   its output states plainly that the bar was not met and that the project is expected to stop
   at Phase 0.

---

### User Story 2 - Grow a library of only-faster GPU operators (Priority: P2)

Once the spike has shown a real win, a developer builds a minimal library of GPU columnar
operators, starting from the single operator that won in Phase 0 and adding operators one at a
time. Each new operator is checked for exact correctness against DuckDB CPU output and
benchmarked against it end to end before the next operator is started. Any operator that does
not beat CPU DuckDB is removed rather than kept or speculatively optimized.

**Why this priority**: The accelerator's value is precisely the set of operators that are
actually faster than the CPU baseline. Adding operators that are not faster dilutes that value
and enlarges the fallback boundary for no gain. This story only exists if User Story 1 passed.

**Independent Test**: For any single operator in the library, run its correctness test against
DuckDB CPU output (including nulls and the integer/float types in scope) and its end-to-end
benchmark against DuckDB CPU. A passing operator is one that is both exactly correct and
faster; a failing one is removed. Each operator is verifiable on its own.

**Acceptance Scenarios**:

1. **Given** a passing Phase 0 result, **When** the library is initialized, **Then** it
   contains exactly the operator that won in Phase 0 and no speculative extras.
2. **Given** a candidate new operator, **When** it is added, **Then** it ships with a
   correctness test against DuckDB CPU output and an end-to-end benchmark against DuckDB CPU,
   and no further operator is started until both are complete.
3. **Given** input columns containing null values and the integer and floating types used by
   the target queries, **When** an operator runs, **Then** its output matches DuckDB CPU output
   exactly for those nulls and types.
4. **Given** a candidate operator whose end-to-end benchmark does not beat CPU DuckDB, **When**
   its benchmark result is recorded, **Then** the operator is removed from the library rather
   than retained or optimized speculatively.

---

### User Story 3 - Write normal SQL and get correct results, GPU or not (Priority: P3)

A DuckDB user loads the extension and writes ordinary SQL. Supported operators and types run
on the GPU automatically with no SQL changes; anything the GPU backend does not support falls
back to DuckDB CPU execution transparently within the same query. The user always gets the
exact result they would have gotten from pure CPU DuckDB, and can force the CPU path entirely
with a setting.

**Why this priority**: This is the eventual product, but it depends on User Story 2 having
produced at least one operator worth routing to the GPU. It is the last gate, not the first.

**Independent Test**: Load the extension on Apple Silicon, run a query mixing supported and
unsupported operators, and confirm the result is identical to pure CPU DuckDB with no
duplicated, missing, or approximate rows; then flip the disable setting and confirm the query
runs entirely on CPU.

**Acceptance Scenarios**:

1. **Given** the extension is loaded, **When** a user runs a query composed of supported
   operators and types, **Then** it routes to the GPU with no SQL changes and returns the exact
   result.
2. **Given** a query that contains an operator or type the GPU backend does not support,
   **When** it runs, **Then** the unsupported portion falls back to DuckDB CPU execution within
   the same query and the result is exactly equivalent to pure CPU execution — no duplicated,
   missing, or approximate rows.
3. **Given** the GPU-disable setting is turned on, **When** any query runs, **Then** it
   executes entirely on the CPU path.
4. **Given** a stock Apple Silicon machine, **When** the extension is built and loaded, **Then**
   it builds and loads successfully.

---

### Edge Cases

- **Integer overflow in aggregation**: When a `SUM` over a group exceeds the range of its
  accumulator type, the GPU result MUST match DuckDB's CPU behavior exactly rather than
  silently wrapping to a different value than the CPU would produce.
- **Nulls in group keys and values**: Grouping and aggregation MUST treat null keys and null
  values exactly as DuckDB CPU does (null as its own group, nulls skipped in `SUM`/`COUNT` per
  DuckDB semantics).
- **Empty and single-group inputs**: Aggregation over zero rows, or over data that collapses
  to a single group, MUST match CPU output.
- **Mixed supported/unsupported within one query (Phase 2)**: A single query where only part
  is GPU-eligible MUST split cleanly with no double-counted or dropped rows at the boundary.
- **Result does not match during the spike**: If the GPU result differs from DuckDB's, the
  spike MUST fail loudly and not report a misleading speedup for a wrong answer.
- **GPU faster but under the bar**: A GPU path that wins but by less than 3× MUST be reported
  as a non-pass, not rounded up to a success.
- **Dataset does not fit comfortably in unified memory**: Behavior when the working set
  approaches available unified memory is out of scope for the spike and is recorded as a
  non-goal for now.

## Requirements *(mandatory)*

### Functional Requirements

**Phase 0 — Validation spike (gating)**

- **FR-001**: The system MUST provide a single script that generates a reproducible dataset of
  a few hundred million rows with integer group keys and numeric values.
- **FR-002**: The system MUST run a `GROUP BY` with `SUM` and `COUNT` in plain CPU DuckDB and
  record its wall-clock time.
- **FR-003**: The system MUST run the same aggregation on the Apple GPU by reading the columns
  out of DuckDB as Arrow, computing on the GPU, and returning the result.
- **FR-004**: The system MUST verify that the GPU result is numerically identical to the DuckDB
  CPU result and fail visibly if it is not.
- **FR-005**: The system MUST report both wall-clock timings and the resulting speedup or
  slowdown, with the Arrow conversion cost included in the GPU timing.
- **FR-006**: The system MUST run on a stock Apple Silicon machine with no NVIDIA hardware and
  no CUDA present.
- **FR-007**: The system MUST state explicitly whether the GPU path beat CPU DuckDB by at least
  3× end to end, and when it did not, state that the project is expected to stop at Phase 0.

**Phase 1 — Columnar operator library (only if Phase 0 passes)**

- **FR-008**: The system MUST begin the operator library with exactly the operator that won in
  Phase 0 and add further operators one at a time.
- **FR-009**: Each operator MUST ship with a correctness test against DuckDB CPU output and an
  end-to-end benchmark against DuckDB CPU output before the next operator is started.
- **FR-010**: Each operator MUST handle null values and the integer and floating types used by
  the target queries, matching DuckDB CPU semantics exactly.
- **FR-011**: The system MUST remove any operator whose end-to-end benchmark does not beat CPU
  DuckDB, rather than retaining or speculatively optimizing it.

**Phase 2 — DuckDB extension with transparent interception (only after Phase 1 has winners)**

- **FR-012**: Loading the extension MUST cause supported queries to route to the GPU with no
  SQL changes.
- **FR-013**: Any unsupported operator or type MUST fall back to DuckDB CPU execution within
  the same query.
- **FR-014**: Fallback MUST never produce duplicated, missing, or approximate rows; a query's
  result MUST be exactly equivalent to pure CPU execution.
- **FR-015**: The system MUST provide a setting that disables GPU execution and forces the CPU
  path.
- **FR-016**: The extension MUST build and load on Apple Silicon.

**Cross-cutting**

- **FR-017**: The system MUST NOT return an approximate or partial result in place of the exact
  one without an explicit caller-supplied flag.
- **FR-018**: Every benchmark MUST measure end to end, including any data-format conversion
  cost, and MUST NOT report kernel-only time as the comparison figure.

### Key Entities *(include if feature involves data)*

- **Benchmark dataset**: A reproducible table of a few hundred million rows, each with an
  integer group key and one or more numeric value columns, generated from a fixed seed so runs
  are comparable.
- **Aggregation result**: The set of (group key → SUM, COUNT) rows produced by the operator,
  used both as the returned answer and as the correctness-comparison artifact against DuckDB.
- **Benchmark record**: For a given operator and dataset, the CPU wall-clock time, the GPU
  end-to-end wall-clock time (conversion included), the resulting ratio, and the pass/fail
  verdict against the baseline.
- **Operator**: A single relational computation offered by the GPU backend, carrying its own
  correctness test and benchmark, and a supported/unsupported status for types and nulls.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer can obtain a go/no-go decision for the whole project by running one
  script on one Apple Silicon machine, with no additional hardware and no manual result
  checking.
- **SC-002**: The spike's reported GPU timing always includes the cost of getting data from
  DuckDB into GPU-computable form; there is no configuration in which the reported figure is
  kernel-time-only.
- **SC-003**: For every reported speedup, the GPU result has been confirmed exactly equal to
  DuckDB's own result for the same query; no speedup is ever reported for a divergent answer.
- **SC-004**: The project proceeds past Phase 0 only when the GPU path beats CPU DuckDB by at
  least 3× end to end; a smaller win produces a documented stop.
- **SC-005**: Every operator retained in the Phase 1 library has a recorded benchmark showing
  it beat CPU DuckDB end to end; no retained operator lacks such a record.
- **SC-006**: In Phase 2, any query a user runs returns a result exactly equal to what pure CPU
  DuckDB returns for that query, whether it ran on the GPU, fell back, or was forced to CPU by
  the setting.

## Assumptions

- The target machine is a stock Apple Silicon Mac with unified memory; performance and
  correctness are always judged against CPU DuckDB on that same machine.
- The benchmark dataset fits comfortably in unified memory; datasets that approach the memory
  limit are out of scope for now (see Non-Goals).
- "A few hundred million rows" is chosen to make the aggregation heavy enough that a real
  difference between CPU and GPU is observable; the exact row count is a tuning detail recorded
  in the plan, not a scope decision.
- Apache Arrow is the interchange between DuckDB and the GPU backend, and its conversion cost
  is always counted in benchmarks.
- The orchestration design (plan interception, fallback, pipelining, memory management) is
  adapted from the hardware-agnostic upper half of an existing GPU-native DuckDB engine; only
  the compute backend and its allocator are original work.
- The specific GPU compute technology (e.g. an Apple-only array framework versus a portable
  layer) is an explicit architecture decision recorded in the plan, not fixed by this spec.

## Non-Goals

- Multi-GPU execution.
- Distributed / multi-node execution.
- GPU-direct storage IO.
- Any NVIDIA or CUDA target.
- Handling working sets that exceed available unified memory.

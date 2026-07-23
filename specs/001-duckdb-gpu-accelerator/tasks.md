# Tasks: DuckDB Apple Silicon GPU Accelerator

**Feature**: `specs/001-duckdb-gpu-accelerator` | **Branch**: `task/task-3-1cc1da`
**Inputs**: plan.md, spec.md, research.md, data-model.md, contracts/spike-cli.md, quickstart.md

## The gate that governs this whole file

Phase 0 (User Story 1) is a hard gate. **No task in Phase 1 or Phase 2 may start until
`spike/bench_groupby.py` prints `verdict=PASS` (correct AND ratio ≥ 3.0).** A `FAIL (under
gate)` result is a documented project stop — Phases 1 and 2 are abandoned, not deferred. This
mirrors Constitution Principle I and is not negotiable at the task level.

Legend: `[P]` = parallelizable (different files, no incomplete dependency). Story labels
`[US1]`/`[US2]`/`[US3]` map to the three user stories in spec.md.

---

## Phase 1: Setup

- [ ] T001 Create `spike/` directory and a `data/` (gitignored) output location per plan.md project structure
- [ ] T002 [P] Add a `spike/README.md` stating the spike is throwaway and gated: PASS unlocks Phase 1, FAIL stops the project
- [ ] T003 Verify the venv satisfies the spike: `duckdb`, `pyarrow`, `mlx`, `numpy` importable and `mlx.core.default_device()` is a GPU (record output in `spike/README.md`)

---

## Phase 2: Foundational (blocking prerequisites for the spike)

- [ ] T004 Fix the run constants (seed=42, rows≈300_000_000, key cardinality) in one place `spike/config.py` so CPU and GPU read identical bytes (data-model.md § BenchmarkDataset)
- [ ] T005 Decide and document the honest-timing contract in `spike/timing.py`: a context manager whose GPU span includes Arrow export + MLX conversion + compute + `mx.eval()` + host materialization (research.md R5, contract G1)

**Checkpoint**: constants and timing helper exist; the spike scripts can be built on them.

---

## Phase 3: User Story 1 — Validation spike decides go/no-go (Priority: P1) 🎯 MVP + GATE

**Goal**: One script produces a verified, honest CPU-vs-GPU number and a PASS/FAIL verdict
against the 3× bar.

**Independent test**: On a stock Apple Silicon machine, `generate_data.py` then
`bench_groupby.py` runs to completion, verifies correctness, and prints the verdict; exit code
is 0 only on PASS. (spec.md US1, contracts/spike-cli.md)

- [ ] T006 [US1] Implement `spike/generate_data.py`: reproducible ~300M-row dataset (INT `key`, INT64 `val_i`, DOUBLE `val_d`, with some NULLs) into a DuckDB file from the fixed seed; print row count + distinct-key count (FR-001, data-model.md)
- [ ] T007 [US1] Implement the CPU baseline in `spike/bench_groupby.py`: `SELECT key, SUM(val_i), SUM(val_d), COUNT(*) FROM t GROUP BY key`, timed with `spike/timing.py` (FR-002)
- [ ] T008 [US1] Implement the GPU path in `spike/bench_groupby.py`: export columns as Arrow, convert to MLX, sort-based group-by (argsort → gather → segment boundaries → segmented SUM + COUNT), `mx.eval()`, materialize back — all inside the GPU timing span (FR-003, FR-005, FR-018, research.md R2/R4)
- [ ] T009 [US1] Implement `spike/verify.py`: sort both results by `key`; assert exact equality on `key`, `count`, and integer `sum_i`; compare float `sum_d` within a documented tolerance; handle NULL key group and skipped-NULL values (FR-004, SC-003, research.md R6)
- [ ] T010 [US1] Wire the verdict in `spike/bench_groupby.py`: `PASS` iff correct AND ratio ≥ gate(3.0); print full benchmark record; on `FAIL (under gate)` also print the explicit "project is expected to stop at Phase 0" sentence; exit non-zero on any FAIL (FR-007, contract G2/G5)
- [ ] T011 [US1] Run the gate end-to-end on this machine; record the actual `cpu_seconds`, `gpu_seconds`, `ratio`, and `verdict` in `spike/README.md`

**🚦 GATE CHECKPOINT**: If T011 is `PASS`, Phase 1 is unlocked. If `FAIL`, stop here, record
the negative result, and do not proceed to Phase 4+.

---

## Phase 4: Backend decision of record (blocks all Phase 1 kernels)

> Only after the Phase 3 gate is PASS.

- [ ] T012 Record the compute-backend decision in a durable ADR `docs/adr/0001-compute-backend.md`: chosen backend (MLX for spike → raw Metal/MPS for the library) with rejected options and reasons (MLX-forever: no relational primitives/strings; Vulkan/Cyfra: MoltenVK layer + JVM boundary), and the revisit trigger (multi-vendor goal / out-of-process service). Mirrors plan.md § Compute Backend Decision; Constitution Principle V. **No Phase 1 kernel task may start before this ADR exists.**

---

## Phase 5: User Story 2 — Library of only-faster operators (Priority: P2)

> Gated on Phase 3 PASS and the T012 ADR. Each operator is a **triple**: implementation +
> correctness-vs-DuckDB + benchmark-vs-DuckDB, and is **removed** if its benchmark fails.

**Goal**: A minimal Metal operator library containing only operators proven faster than CPU
DuckDB end to end.

**Independent test**: For any single operator, its correctness test passes against DuckDB CPU
(nulls + int/float types) and its end-to-end benchmark beats CPU DuckDB; a failing benchmark
removes it. (spec.md US2)

### Foundation for the library

- [ ] T013 [US2] Scaffold `src/kernels/` and `src/allocator/` (unified-memory allocator, no discrete-pool tiering) per plan.md; original work only, no Sirius compute layer (Principle III)
- [ ] T014 [US2] Establish `tests/correctness/` and `tests/benchmark/` harnesses that diff any operator output against DuckDB CPU and time it end to end (Arrow cost included) (FR-009, FR-018)

### Operator 1 — the Phase 0 winner, reimplemented as a Metal kernel

- [ ] T015 [US2] Implement `group_by_sum_count` Metal kernel in `src/kernels/group_by.mm` (hash-based aggregation; the spike proved the operator, Metal raises the ceiling)
- [ ] T016 [P] [US2] Correctness test `tests/correctness/test_group_by.py` vs DuckDB CPU: exact int SUM/COUNT, tolerance float SUM, NULL key + skipped-NULL semantics, integer-overflow parity (FR-010, spec edge cases)
- [ ] T017 [US2] Benchmark `tests/benchmark/bench_group_by.py` vs DuckDB CPU end to end; **if it does not beat CPU, mark the operator `removed` and delete the kernel** (FR-011, data-model.md § Operator state transitions)

### Operator N — template to repeat one operator at a time

- [ ] T018 [US2] For each subsequent candidate operator (e.g. `filter`, then `hash_join`), added strictly one at a time: (a) implement the kernel in `src/kernels/<op>.mm`, (b) [P] correctness test vs DuckDB CPU incl. nulls + int/float types, (c) benchmark vs DuckDB CPU end to end, (d) **remove the operator if the benchmark subtask fails**. Do not start operator N+1 until N's triple is complete (FR-008, FR-009, FR-011)

**Checkpoint**: library contains only operators with a recorded end-to-end win.

---

## Phase 6: User Story 3 — DuckDB extension with transparent interception (Priority: P3)

> Gated on Phase 5 having at least one `retained` operator.

**Goal**: Loading the extension routes supported SQL to the GPU and falls back to CPU for the
rest, always returning results exactly equal to pure CPU DuckDB.

**Independent test**: Load on Apple Silicon, run a mixed supported/unsupported query → result
identical to pure CPU with no dup/missing/approx rows; flip the disable setting → all-CPU.
(spec.md US3)

- [ ] T019 [US3] Scaffold the DuckDB C++ extension in `src/extension/` from the extension template; builds and loads on Apple Silicon (FR-016)
- [ ] T020 [US3] Adapt the Sirius-style optimizer-hook interception in `src/extension/interception.cpp` to route supported subplans to the retained GPU operators (Principle III, plan.md § Integration Seam)
- [ ] T021 [US3] Implement per-query CPU fallback in `src/extension/fallback.cpp`: any unsupported operator/type routes to DuckDB CPU within the same query, with no duplicated/missing/approximate rows at the boundary (FR-013, FR-014, spec edge case)
- [ ] T022 [P] [US3] Implement the `gpu_execution` disable setting that forces the all-CPU path (FR-015)
- [ ] T023 [US3] SQL-logic tests in `tests/correctness/test_extension.py`: mixed-operator query equals pure CPU; forced-CPU equals pure CPU; overflow/null parity across the fallback boundary (FR-012, FR-014, SC-006)
- [ ] T024 [US3] Enforce Principle IV: no approximate/partial result is returned without an explicit flag; add a test asserting this (FR-017)

**Checkpoint**: normal SQL is transparently accelerated with exact-result fallback.

---

## Phase 7: Polish & cross-cutting

- [ ] T025 [P] Confirm every retained operator carries a committed BenchmarkRecord showing its end-to-end win; delete any operator lacking one (Principle II, SC-005)
- [ ] T026 [P] Sweep for dead/speculative operators and kernels not proven faster; remove them (.roborev.toml review weighting)
- [ ] T027 [P] Update README.md § Status with the Phase 0 verdict and, if applicable, the retained-operator list
- [ ] T028 Run the full correctness + benchmark suites on Apple Silicon and record results

---

## Dependencies & story completion order

- **Setup (T001–T003)** → **Foundational (T004–T005)** → **US1 / Phase 3 (T006–T011)**.
- **🚦 Gate (T011)** blocks everything below. PASS unlocks; FAIL stops the project.
- **T012 ADR** blocks all US2 kernel tasks (T015+).
- **US2 (T013–T018)** depends on the gate + ADR. Operators are strictly sequential (one triple
  at a time); within a triple, the correctness test `[P]` can run alongside the benchmark once
  the kernel exists.
- **US3 (T019–T024)** depends on US2 producing ≥1 retained operator.
- **Polish (T025–T028)** last.

## Parallel opportunities

- T002 alongside T001/T003 (docs vs env check).
- Within each operator triple, the `[P]` correctness test parallels the benchmark task.
- T022 (disable setting) parallels T020/T021 interception/fallback work.
- Polish T025–T027 are mutually `[P]`.

## Implementation strategy

- **MVP = User Story 1 only.** The spike is the deliverable that earns the right to build
  anything else. Ship it, read the verdict, then decide.
- **Incremental after the gate**: one operator at a time (US2), each proven faster or removed,
  before the extension (US3) wraps the winners.

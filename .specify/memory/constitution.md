<!--
SYNC IMPACT REPORT
==================
Version change: (uninitialized template) → 1.0.0
Bump rationale: Initial ratification. First concrete constitution replacing the
  placeholder template; MAJOR baseline per semantic-versioning policy below.

Principles defined (all new):
  I.   Validation Before Construction
  II.  Benchmark-Gated Increments
  III. Reuse Orchestration, Replace Only Compute
  IV.  Fallback Is Mandatory; Silent Wrong Answers Forbidden
  V.   Portability Is a Decision, Not a Default

Added sections:
  - Additional Constraints (baseline, hardware, data interchange)
  - Development Workflow (gate order, review discipline)
  - Governance

Removed sections: none (template placeholders replaced wholesale).

Key project constant:
  - GATE MARGIN = 3× end-to-end speedup vs CPU DuckDB (set in Principle I).

Templates requiring updates:
  ✅ .specify/templates/plan-template.md   — Constitution Check gates align (verified)
  ✅ .specify/templates/spec-template.md   — no principle-driven mandatory sections added
  ✅ .specify/templates/tasks-template.md  — benchmark/fallback task types already expressible
  ✅ README.md                             — principles + 3× gate already documented

Follow-up TODOs: none. Ratification date set to first constitution commit.
-->

# DuckDB Apple Silicon GPU Accelerator Constitution

## Core Principles

### I. Validation Before Construction

No columnar kernel library, no DuckDB extension, and no Metal shaders MAY be written until a
throwaway spike proves that GPU execution of at least one relational operator beats CPU DuckDB
on the same Apple Silicon machine by **at least 3× end to end**. "End to end" includes every
cost the production path would pay — Arrow conversion, buffer handoff, and result
materialization — not kernel time in isolation. If the spike does not clear the 3× bar, the
project HALTS at Phase 0 and reports the negative result; it does not proceed to build on the
hope that later engineering will recover the margin.

CPU DuckDB is the baseline every stage is measured against. Rationale: Apple Silicon uses
unified memory, so the CPU engine already runs in the same memory as the GPU with no
host-to-device transfer to amortize. The usual GPU advantage — hiding a slow operator behind a
transfer that CPU code would also pay — does not exist here, which is exactly why the bar is
high and validation comes first.

### II. Benchmark-Gated Increments

Every operator added to the library MUST be benchmarked against CPU DuckDB on representative
data before the next operator is started. Benchmarks MUST measure end to end, including any
data-format conversion cost. An operator that does not beat the CPU baseline is DROPPED, not
optimized speculatively and not retained "for completeness." Speculative optimization of an
operator that has not first demonstrated a win is out of scope. Rationale: the accelerator's
value is the set of operators that are actually faster; carrying slower ones dilutes it and
adds fallback-boundary surface for no gain.

### III. Reuse Orchestration, Replace Only Compute

Plan interception, CPU fallback, pipelining, scheduling, and memory management are ADAPTED
from the hardware-agnostic upper half of an existing GPU-native DuckDB engine (the reference
design). Original engineering is CONFINED to the compute backend — the columnar kernels — and
its memory allocator. The CUDA/cuDF-locked compute layer of the reference engine MUST NOT be
reused. Rationale: orchestration is a solved, portable problem; the project's novel risk and
value both live in the Apple-GPU compute path, so that is where original work is spent.

### IV. Fallback Is Mandatory; Silent Wrong Answers Forbidden

Any operator or data type the GPU backend does not support MUST route to DuckDB CPU execution.
The GPU path is never the only way to obtain a correct answer. The system MUST NOT return an
approximate, partial, duplicated, or missing-row result in place of the exact one without an
explicit, caller-supplied flag. Fallback within a single query MUST preserve exact result
equivalence with pure CPU execution. Rationale: a query engine that can silently be wrong is
unusable regardless of speed; correctness is not tradeable against performance.

### V. Portability Is a Decision, Not a Default

Whether the compute backend targets Apple GPUs only (Metal/MLX) or multiple vendors through a
portable layer (Vulkan or SPIR-V) is an EXPLICIT, RECORDED architecture decision made in the
plan and justified against the project's actual goal. It MUST NOT be assumed in either
direction, and MUST NOT drift implicitly through library choices made for other reasons. The
recorded decision MUST list the rejected options with reasons. Rationale: portability is a
large, load-bearing cost that changes the whole backend design; deciding it by accident is how
projects acquire a translation layer or a runtime boundary nobody chose.

## Additional Constraints

- **Baseline of record.** CPU DuckDB on the same machine is the sole performance and
  correctness baseline. No comparison against a different engine, a different machine, or a
  synthetic kernel-only timing may substitute for it.
- **Gate margin.** The Phase 0 continuation gate is **3× end to end**. Changing this number is
  a constitutional amendment (see Governance), not a plan-level or code-level decision.
- **Target hardware.** Stock Apple Silicon, unified memory. No NVIDIA hardware and no CUDA are
  in scope. Memory management relies on unified memory; the reference engine's discrete-GPU
  spilling and GPU-direct IO machinery MUST NOT be ported wholesale.
- **Data interchange.** Apache Arrow between DuckDB and the GPU backend, zero-copy where buffer
  layouts line up. Arrow conversion cost is counted in every benchmark, always.
- **Numerical correctness.** Every operator is validated for exact numerical equality against
  DuckDB CPU output on the same machine, including null handling and integer overflow behavior
  in SUM-style aggregations, before its benchmark result counts.

## Development Workflow

- **Phase gating is strict.** Phase 1 (operator library) MUST NOT begin until the Phase 0
  spike has produced a passing ≥3× benchmark. Phase 2 (DuckDB extension) MUST NOT begin until
  Phase 1 has at least one operator that beat the baseline. The compute-backend decision
  (Principle V) MUST be recorded before any Phase 1 kernel is written.
- **Every operator is a triple.** Implementation, a correctness test versus DuckDB CPU output,
  and a benchmark versus DuckDB CPU output. An operator whose benchmark fails the bar is marked
  for removal, not iteration.
- **Review discipline.** Changes are reviewed with weight on: numerical correctness versus
  DuckDB CPU output (nulls, overflow), presence of a working CPU fallback for every unsupported
  operator or type, benchmark honesty (conversion cost included), memory safety in kernels and
  the Arrow/GPU buffer handoff, and removal of dead or speculative operators.

## Governance

This constitution supersedes ad-hoc practice for this project. When a principle and a
convenience conflict, the principle wins or the constitution is formally amended first — it is
not silently overridden in code or plan.

**Amendment procedure.** Any change to a principle, to the 3× gate margin, or to the phase
gating MUST be made as an edit to this file with a Sync Impact Report and a version bump, and
MUST propagate to dependent templates (`plan-template.md`, `spec-template.md`,
`tasks-template.md`) and to `README.md` where they reference the changed rule.

**Versioning policy (semantic).**
- MAJOR: backward-incompatible governance or principle removal/redefinition, or a change to the
  gate margin or the mandatory-fallback rule.
- MINOR: a new principle or section, or materially expanded guidance.
- PATCH: clarifications, wording, and non-semantic refinements.

**Compliance review.** Every review and merge verifies compliance with Principles I–V.
Complexity that is not justified against the project's actual goal (Principle V) or not proven
faster than CPU DuckDB (Principles I–II) is removed. Runtime guidance for contributors lives in
`README.md` and `.roborev.toml`.

**Version**: 1.0.0 | **Ratified**: 2026-07-23 | **Last Amended**: 2026-07-23

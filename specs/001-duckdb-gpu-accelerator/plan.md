# Implementation Plan: DuckDB Apple Silicon GPU Accelerator

**Branch**: `task/task-3-1cc1da` | **Date**: 2026-07-23 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-duckdb-gpu-accelerator/spec.md`

## Summary

Answer, cheaply and first, whether the Apple GPU can beat CPU DuckDB at a heavy relational
operator on the same unified-memory machine; then, only if it can by ≥3× end to end, grow a
library of only-faster GPU columnar operators and finally wrap them in a DuckDB extension that
routes supported work to the GPU and falls back to CPU for everything else. The technical crux
is the compute-backend choice and the DuckDB↔GPU seam; both are recorded here.

## Technical Context

**Language/Version**: Python 3.12 for the Phase 0 spike (fastest path to a DuckDB↔Arrow↔GPU
loop); C++17 (DuckDB extension ABI) for Phase 2. Phase 1 kernel language depends on the
backend decision below — Python/MLX first, then Objective-C++/Metal Shading Language.

**Primary Dependencies**: DuckDB 1.5.x (CPU baseline + Arrow export), Apache Arrow (pyarrow 25
for the spike), MLX 0.32 (Phase 0 GPU compute), NumPy (result comparison). Phase 1+: Metal /
Metal Performance Shaders via the system framework. DuckDB C++ extension template for Phase 2.

**Storage**: DuckDB database file for the generated benchmark dataset; regenerable from a fixed
seed, not committed (`.gitignore`d).

**Testing**: Numerical-equality assertions of GPU output against DuckDB CPU output on the same
machine; wall-clock benchmarks with Arrow conversion counted in the GPU figure. `pytest` for
the spike/library; DuckDB's SQL-logic test harness for the Phase 2 extension.

**Target Platform**: Stock Apple Silicon (unified memory). No NVIDIA, no CUDA. Developed/measured
on Apple M-series (this machine: M5 Pro, 48 GB unified).

**Project Type**: Two-stage — a throwaway feasibility spike (single script), then, gated on it,
a compute library plus a native DuckDB extension.

**Performance Goals**: Phase 0 GPU path beats CPU DuckDB by **≥3× end to end** (Arrow conversion
included) on the `GROUP BY … SUM, COUNT` benchmark. Phase 1: each retained operator beats CPU
DuckDB end to end.

**Constraints**: Correctness is never traded for speed; every GPU result must equal DuckDB's
exactly (nulls, integer-overflow semantics). Benchmarks are always end to end, never
kernel-only. Working sets that exceed unified memory are out of scope for now.

**Scale/Scope**: Benchmark dataset of a few hundred million rows (target ~300M) with an integer
group key of moderate cardinality and one or more numeric value columns.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | How this plan complies |
|---|---|
| **I. Validation before construction** | Phase 0 is a single throwaway script. No kernel library, no extension, no Metal shaders until it clears ≥3× end to end. A sub-3× result is a documented stop, not a reason to keep building. |
| **II. Benchmark-gated increments** | Phase 1 adds operators one at a time, each with an end-to-end benchmark (Arrow cost included) versus CPU DuckDB. Losers are removed. |
| **III. Reuse orchestration, replace only compute** | Orchestration (interception, fallback, pipelining, memory) is adapted from the hardware-agnostic upper half of Sirius. Original work is confined to the compute backend + its allocator. Its CUDA/cuDF compute layer is not reused. |
| **IV. Fallback mandatory; no silent wrong answers** | Phase 2 falls back to CPU for unsupported operators/types within the same query; results stay exactly equal to pure CPU. No approximate/partial result without an explicit flag. |
| **V. Portability is a decision** | The compute-backend decision is recorded below with rejected options and reasons (MLX → Metal, Vulkan/Cyfra rejected for now). It is not assumed. |

**Verdict**: PASS. No violations to justify; Complexity Tracking left empty.

## Compute Backend Decision (Constitution Principle V)

**Decision**: **MLX for the Phase 0 spike, then raw Metal / Metal Performance Shaders for the
Phase 1 library.** Apple-only. Recorded and justified; not assumed.

**Why**: It keeps everything native to the eventual C++ DuckDB extension and inside unified
memory, with no translation layer and no cross-runtime boundary. MLX gets a sort-based
group-by standing up in Phase 0 with zero shader code (it ships argsort, gather, cumsum, and
segmented reductions); Metal gives the performance ceiling for Phase 1 once MLX has proven an
operator is worth the engineering.

**Rejected options** (recorded per Principle V):

- **Option A alone — MLX for everything.** Rejected as the *long-term* backend: it is an
  array/ML framework with no relational primitives and no string columns; it is a prototyping
  tool, not a library foundation. Kept for Phase 0 only.
- **Option C — Vulkan/SPIR-V (e.g. Cyfra), portable.** Rejected *for now*. Portability across
  NVIDIA/AMD/Intel is not a stated primary goal of this Apple-specific project. Costs: a
  MoltenVK translation layer on Apple instead of native Metal, a pre-release dependency, and —
  with Cyfra — a Scala/JVM runtime that a native C++ DuckDB extension would have to cross. That
  JVM boundary is a significant mark against it for an in-process Phase 2 extension.
- **Revisit trigger**: reopen Option C only if the project goal changes to a *multi-vendor*
  accelerator, or if an out-of-process kernel service (where the JVM boundary is cheaper)
  becomes the chosen architecture.

## Integration Seam

- **Reference design**: the hardware-agnostic upper half of the **Sirius** GPU-native DuckDB
  engine — plan interception, CPU fallback, pipelining, memory management — adapted, not copied.
  Its compute layer (CUDA/cuDF-locked, Apple-incompatible) is **not** reused.
- **Phase 0 seam (simple)**: pull columns out of DuckDB as Arrow *outside* any extension,
  compute on the GPU, return Arrow. No optimizer hooks.
- **Phase 2 seam (full)**: optimizer-hook interception inside the loaded extension routes
  supported subplans to the GPU and leaves the rest to CPU. This is a Phase 2 concern only.

## Memory & Data Interchange

- **Unified memory throughout.** No discrete GPU pool to tier, no host-to-device staging, so
  the reference engine's spilling and GPU-direct-IO machinery collapses to plain memory-mapped
  access. It is **not** ported wholesale.
- **Apache Arrow** is the DuckDB↔GPU interchange, zero-copy where buffer layouts line up. The
  Arrow conversion cost is **counted in every benchmark** — there is no kernel-only comparison
  figure anywhere.

## Project Structure

### Documentation (this feature)

```text
specs/001-duckdb-gpu-accelerator/
├── plan.md              # This file (/speckit-plan output)
├── spec.md              # Feature specification (/speckit-specify output)
├── research.md          # Phase 0 output — backend + dataset + group-by strategy
├── data-model.md        # Phase 1 output — dataset, result, benchmark record
├── quickstart.md        # Phase 1 output — how to run the spike
├── contracts/
│   └── spike-cli.md     # Phase 0 spike command + output contract
├── checklists/
│   └── requirements.md  # Spec quality checklist (/speckit-specify output)
└── tasks.md             # /speckit-tasks output (NOT created here)
```

### Source Code (repository root)

```text
spike/                       # Phase 0 — throwaway, deleted or archived if it fails the gate
├── generate_data.py         # reproducible ~300M-row dataset from a fixed seed
├── bench_groupby.py         # CPU DuckDB vs Apple-GPU (MLX) head-to-head, 3x verdict
└── verify.py                # numerical-equality check of GPU result vs DuckDB

# The trees below are created only after Phase 0 passes the gate:
src/
├── kernels/                 # Phase 1 — Metal columnar operators (original work)
├── allocator/               # Phase 1 — unified-memory allocator (original work)
└── extension/               # Phase 2 — DuckDB C++ extension, interception + fallback

tests/
├── correctness/             # per-operator equality vs DuckDB CPU
└── benchmark/               # per-operator end-to-end timing vs DuckDB CPU
```

**Structure Decision**: Start with a self-contained `spike/` directory of Python scripts — the
only code that exists until the gate is cleared. `src/` and `tests/` are deliberately *not*
scaffolded now; creating them before Phase 0 passes would violate Principle I. They appear in
the tree above as the committed-to layout for after the gate, nothing more.

## Complexity Tracking

> No Constitution Check violations. Section intentionally empty.

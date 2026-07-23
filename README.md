# DuckDB GPU acceleration on Apple Silicon

An experiment in offloading DuckDB analytical operators to the Apple Silicon GPU, with
transparent fallback to DuckDB's CPU engine for anything unsupported.

The project is **validation-gated**. DuckDB's CPU engine is already fast on Apple Silicon:
it is vectorized and runs in unified memory, so there is no host-to-device transfer to
amortize away. Whether moving relational operators onto the GPU beats that baseline is an
open question. Phase 0 exists to answer it cheaply. If the answer is no, the project stops
at Phase 0 and reports the negative result.

## Governing principles

1. **Validation before construction.** No kernel library, no DuckDB extension, and no Metal
   shaders until a throwaway spike proves GPU execution of at least one relational operator
   beats CPU DuckDB on the same machine by a margin worth the added complexity.
2. **Benchmark-gated increments.** Every operator is benchmarked against CPU DuckDB on
   representative data before the next one is started. Benchmarks measure end to end,
   including data-format conversion, never kernel time in isolation. An operator that does
   not beat the baseline is dropped, not speculatively optimized.
3. **Reuse orchestration, replace only compute.** Plan interception, fallback, pipelining,
   scheduling, and memory management are adapted from the hardware-agnostic upper half of
   an existing GPU-native DuckDB engine. Original work is confined to the compute backend
   and its allocator.
4. **Fallback is mandatory; silent wrong answers are forbidden.** Unsupported operators and
   types route to DuckDB CPU execution. No approximate or partial result is ever returned
   in place of the exact one without an explicit flag.
5. **Portability is a decision, not a default.** Apple-only (Metal/MLX) versus cross-vendor
   (Vulkan/SPIR-V) is an explicit, recorded architecture decision, justified against the
   project's actual goal.

## Roadmap

### Phase 0 — Validation spike (gating deliverable)

A single script that runs one heavy relational operator on both CPU DuckDB and the Apple GPU
over the same data, and prints a head-to-head timing.

- Reproducible dataset of a few hundred million rows: integer group keys, numeric values.
- `GROUP BY` with `SUM` and `COUNT` in plain CPU DuckDB, wall-clock recorded.
- The same aggregation on the GPU — columns pulled out of DuckDB as Arrow, computed on the
  GPU, returned.
- GPU result verified numerically identical to the DuckDB result.
- Both timings reported, plus speedup or slowdown, with Arrow conversion cost counted in the
  GPU number.
- Runs on a stock Apple Silicon machine. No NVIDIA hardware, no CUDA.
- If the GPU path does not clear the bar, the output says so plainly and the project stops.

### Phase 1 — Columnar operator library (only if Phase 0 passes)

A minimal library of GPU columnar operators, seeded with whichever operator won Phase 0 and
grown one operator at a time.

- Each operator ships with a correctness test against DuckDB CPU output and a benchmark
  against it.
- Nulls and the integer and floating types used by the target queries are handled.
- Operators that do not beat CPU DuckDB are removed.

### Phase 2 — DuckDB extension with transparent interception

- Loading the extension routes supported queries to the GPU with no SQL changes.
- Unsupported operators and types fall back to CPU execution within the same query.
- Fallback never produces duplicated, missing, or approximate rows.
- A setting disables GPU execution and forces the CPU path.
- Builds and loads on Apple Silicon.

**Out of scope:** multi-GPU, distributed execution, GPU-direct storage IO, and any
NVIDIA/CUDA target.

## Technical approach

**Integration seam.** The hardware-agnostic upper half of the Sirius GPU-native DuckDB
engine serves as the reference design for plan interception, CPU fallback, pipelining, and
memory management. Its compute layer is CUDA/cuDF-locked and Apple-incompatible, so it is
not reused. For Phase 0 the seam is simpler still: pull columns as Arrow outside any
extension, compute on the GPU, return Arrow. Optimizer-hook interception is a Phase 2
concern.

**Compute backend.** Three options were weighed:

| Option | Reach | Assessment |
| --- | --- | --- |
| **A — MLX** | Apple only | Fastest to prototype. Ships argsort, gather, cumsum, and segmented reductions — enough to assemble a sort-based group-by with no shader code. Direct unified-memory ergonomics. Weak beyond prototyping: it is an array/ML framework with no relational primitives and no string columns. |
| **B — Raw Metal / MPS** | Apple only | Highest performance ceiling, most engineering. The right target for the eventual library, once MLX has proven an operator is worth it. |
| **C — Vulkan/SPIR-V** (e.g. Cyfra) | Portable | One kernel set targets Apple (via MoltenVK), NVIDIA, AMD, and Intel. Justified only if cross-vendor portability is a primary goal. Costs: a translation layer on Apple rather than native Metal, a pre-release dependency, and — with Cyfra — a Scala/JVM runtime that a native C++ DuckDB extension would have to cross. That JVM boundary is a significant mark against it for an in-process Phase 2 extension; less so for an out-of-process kernel service. |

Default unless portability becomes a stated primary goal: **MLX for the Phase 0 spike, then
raw Metal for the Phase 1 library** — it keeps everything native to the DuckDB extension and
inside unified memory. Option C is revisited only if the goal shifts to a multi-vendor
accelerator.

**Memory.** Unified memory throughout. There is no discrete GPU pool to tier and no
host-to-device staging, so the reference engine's spilling and GPU-direct IO machinery
collapses to plain memory-mapped access. It is not ported wholesale.

**Data interchange.** Apache Arrow between DuckDB and the GPU backend, zero-copy where
buffer layouts line up. Arrow conversion cost is counted in every benchmark.

**Testing.** Every operator is validated for numerical equality against DuckDB CPU output
and benchmarked against it on the same machine.

## Development

Specs, plan, and task breakdown are managed with
[GitHub Spec Kit](https://github.com/github/spec-kit).

Commits are reviewed in the background with `roborev` (`roborev init` installs the
post-commit hook). Review guidelines live in `.roborev.toml` and weight numerical
correctness against DuckDB CPU output, null handling, integer overflow in `SUM`-style
aggregations, fallback coverage, benchmark honesty, memory safety in the Arrow/GPU buffer
handoff, and dead speculative operators.

## Status

**Phase 0 ran on 2026-07-23 (Apple M5 Pro, 48 GB) and FAILED the gate — the project stops
here, by design.**

On a 300M-row `GROUP BY … SUM, COUNT`, CPU DuckDB took **0.83 s**; the Apple GPU path (MLX,
end to end with Arrow conversion) took **9.84 s** — a **0.08×** result, i.e. the GPU is ~12×
slower. The GPU answer is numerically exact; it is simply not fast enough, and does not come
close to the 3× bar.

Why: on unified memory there is no host-to-device transfer to amortize, so the CPU engine —
already sitting on the data — is hard to beat. Just moving the columns out of DuckDB into
GPU-computable form costs ~3.4 s, about **4× the entire CPU aggregation**, before any GPU
compute happens. Even a hypothetical zero-cost kernel would lose. This is exactly the risk the
constitution's Principle I was written to catch cheaply — and it did, in a throwaway spike, at
the cost of one afternoon rather than a kernel library and a C++ extension.

Two follow-on findings, both measured (see [`spike/README.md`](./spike/README.md)):

- **Bigger data does not help — it hurts.** Across 2M → 300M rows the ratio falls monotonically
  (0.59× → 0.08×). CPU scales linearly; the GPU path scales worse (O(n·log n) sort + O(n) Arrow
  conversion), so there is no fixed overhead for large data to amortize.
- **The GPU wins on arithmetic intensity, not size.** A `GROUP BY … SUM` is memory-bound (~one
  add per row). The Apple GPU pulls ahead only on compute-bound, high-FLOP-per-byte work —
  matrix algebra, PCA/SVD, ML, FFTs — where the one-time Arrow handoff is amortized over many
  FLOPs. That is a numerical layer *on top of* DuckDB, not a replacement for its relational
  operators, and would be a different project with its own spike.

Full analysis and reproduction steps: [`spike/README.md`](./spike/README.md). Phases 1–2 are
not started.

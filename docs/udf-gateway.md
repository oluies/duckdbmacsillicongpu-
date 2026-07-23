# Design note: a GPU numerical UDF gateway for DuckDB (MLX and/or Cyfra)

**Status**: exploratory design, driven by the Phase 0 spikes. Not a commitment.

## Why this, and not the original accelerator

The Phase 0 spikes settled two things:

1. **Relational offload loses.** GPU `GROUP BY … SUM, COUNT` is ~12× slower than CPU DuckDB and
   gets worse with scale — a memory-bound operator on data DuckDB already holds in unified
   memory ([`../spike/README.md`](../spike/README.md)).
2. **Compute-bound numerical offload wins — conditionally.** PCA/matrix algebra on the GPU beats
   CPU by ~5.5× **once the matrix is resident and reused across many operations**; a single
   op per Arrow load is only ~1.3× ([`../spike/PCA.md`](../spike/PCA.md)).

So the interesting artifact is not a relational accelerator but a **numerical UDF gateway**: SQL
stays in DuckDB; heavy linear-algebra / ML kernels run on the GPU and return results, with the
matrix kept resident across calls so the extraction cost amortizes. This is a different project
than Phases 1–2 in the spec, and it would get its own constitution-style validation.

## The one design constraint that dominates everything

> **Residency or bust.** The spike's decisive number: a one-shot GPU op is ~1.3× (extraction-
> bound), a load-once-compute-many session is ~5.5×. The gateway is only worth building if it
> keeps GPU-side state *between* UDF calls. A stateless "Arrow in → GPU → Arrow out" UDF re-pays
> the ~0.74 s load every call and never clears the 3× bar.

Concretely the gateway needs a **handle/session model**: `gpu_load(columns) -> handle`, then
`gpu_pca(handle, k)`, `gpu_matmul(handle, ...)`, `gpu_knn(handle, ...)` operating on the
resident array, and `gpu_free(handle)`. The heavy Arrow→GPU move happens once per handle.

## DuckDB integration seam

DuckDB supports **Arrow-vectorized Python UDFs** (`con.create_function(name, fn, args, ret,
type="arrow")`) and C++ UDFs — both in-process, no network hop (per the DuckDB UDF article).
Verified locally: an `type="arrow"` UDF receives a pyarrow array and returns one. That is the
handoff the gateway rides:

- A UDF receives an Arrow batch (or a table handle) and returns Arrow — zero-copy where layouts
  line up, which is the same seam the spikes used.
- Fallback (constitution Principle IV): any unsupported dtype/op or absent GPU routes to a
  NumPy/BLAS CPU implementation, exact results either way.

## Two backend options

### Option A — MLX, in-process (Apple-only). Recommended default.

What the spikes already use. Python/C++ UDF → MLX on the Apple GPU, native Metal, unified
memory, no extra process or serialization boundary. Residency is trivial: keep `mx.array`
handles in the UDF module's state.

- **Pros**: fastest path (it's the measured 5.5× path), native, no IPC, no JVM, keeps everything
  in one address space. Directly reuses `bench_pca.py`.
- **Cons**: Apple-only; MLX is an array framework (you assemble kernels from primitives); no
  cross-vendor story.

> **Transport note.** An out-of-process backend needs a wire between DuckDB and the service.
> NATS request-reply is a clean, language-agnostic option (and reaches a JVM/Cyfra worker
> without the in-process JVM boundary) — see [`nats-gateway.md`](./nats-gateway.md). Key move:
> send only the *command* over NATS (a query/kernel ref + params) and have the worker load the
> data **out of band** — ideally a GPU-enabled DuckDB sidecar that runs the query locally, so no
> bulk data crosses the bus. The residency lesson still holds (keep the worker's session
> resident), but the control hop is data-size-independent.

### Option B — Cyfra, out-of-process (portable). Revisit trigger, not default.

[Cyfra](https://github.com/ComputeNode/cyfra) compiles a **Scala 3 DSL → SPIR-V → Vulkan**
(LWJGL), running on Apple (via MoltenVK), NVIDIA, AMD, Intel. A DuckDB C++/Python UDF would
marshal Arrow to a **Cyfra JVM kernel service** (Arrow IPC / Flight / shared memory), which runs
SPIR-V kernels and returns Arrow.

- **Pros**: one kernel set targets every vendor — the only reason to take this on is if
  cross-vendor portability becomes a primary goal (constitution Principle V). Out-of-process, so
  the JVM boundary the constitution flagged for an *in-process* extension is much cheaper here.
- **Cons**:
  - **Maturity**: `0.1.0-RC1`, pre-release (184 stars). Betting a data path on an RC.
  - **The IPC hop fights the one constraint that matters.** The spike shows the handoff *is* the
    bottleneck; adding a process boundary + Arrow serialization between DuckDB and the JVM makes
    residency even more essential — the matrix must live inside the Cyfra service across calls,
    or you re-pay extraction *and* IPC every op.
  - On Apple it runs Vulkan-through-MoltenVK, i.e. a translation layer instead of the native
    Metal that MLX already gives for free.

## Recommendation

- **Build the gateway on MLX in-process first** (Option A), with a handle/residency model, a
  small kernel set proven by spike (Gram/PCA, matmul, then distance/kNN), each measured
  load-once-compute-many against CPU with the honest end-to-end method, and a CPU fallback.
- **Keep Cyfra as a recorded Option C**, activated only if the goal becomes multi-vendor. If it
  is, the honest first step is a spike of its own: measure DuckDB→Cyfra Arrow marshaling cost
  and whether a resident-matrix Cyfra service clears 3× on the same PCA workload — the same bar,
  the same method as [`../spike/PCA.md`](../spike/PCA.md).

## Candidate gateway kernels (all compute-bound, all residency-friendly)

| kernel | why it fits (FLOPs/byte) |
|---|---|
| PCA / SVD / covariance | O(N·D²) Gram + factorization; reused across components/iterations |
| dense matmul / linear layers | O(N·D·K); the measured 5.5× case |
| kNN / similarity search | O(N·M·D) distance matrix; heavy reuse of resident vectors |
| k-means / clustering | iterative, resident points reused every iteration |
| FFT / convolutions | high reuse, GPU-native |

Explicitly **not** in scope: relational operators (GROUP BY, join, filter) — the first spike
showed those belong on the CPU.

## Next step

A gateway prototype spike: `gpu_load` + `gpu_pca`/`gpu_matmul` Arrow UDFs on MLX with a handle
table, benchmarked as a multi-call session against a CPU-only equivalent, end to end. If it
holds the ~5× the raw spike showed once wrapped as UDFs, the gateway is worth building; if the
UDF/marshaling overhead erodes it below 3×, that is a documented stop — same discipline as
Phase 0.

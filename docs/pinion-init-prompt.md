# Pinion — repository init prompt

Paste the block below into a fresh Claude Code session opened in an **empty directory** named
`pinion/`. It bootstraps the new repo with the same tooling as this project (Spec Kit + roborev
+ a uv venv) and seeds a constitution that starts from *this* project's validated findings, so
Pinion does not re-litigate what the spikes already settled.

> **What Pinion is** (one line for the agent): a GPU numerical-compute gateway for DuckDB —
> heavy, compute-bound, residency-reused kernels (PCA/SVD, matmul, kNN, clustering, FFT) run on
> the GPU via a worker; DuckDB keeps the SQL. Commands travel over NATS; bulk data does not.

---

```
Bootstrap this repository as "Pinion", a GPU numerical-compute gateway for DuckDB. Set up the
same tooling as my previous project and seed a constitution from its findings. Do all of this,
committing at the end:

1. Git: initialize a repo if one does not exist. Work on a branch, not main.

2. Spec Kit: run `specify init --here --integration claude --script sh`. (Skills install as
   /speckit-constitution, /speckit-specify, /speckit-plan, /speckit-tasks — note the hyphens.)

3. Python env with uv: create a venv and install the deps the gateway spikes need:
     uv venv --python 3.12 .venv
     uv pip install --python .venv/bin/python duckdb pyarrow mlx numpy nats-py pytest
   Smoke-test that duckdb, pyarrow, mlx, numpy, and nats import and that
   `mlx.core.default_device()` is a GPU; record the output.

4. .gitignore: ignore `.venv/`, `__pycache__/`, `*.pyc`, `*.duckdb`, `*.duckdb.wal`, `*.parquet`,
   `data/`, and `/.roborev/`.

5. roborev: run `roborev init`, then write `.roborev.toml` with agent = "claude-code" and review
   guidelines weighting, for THIS project:
     - Benchmark honesty: GPU/worker timings measure END TO END — the worker's local DuckDB→GPU
       load, the compute, and any control round-trip — never kernel-only. Single-op vs
       resident-session numbers must both be reported (residency is where the win lives).
     - Arithmetic intensity as the admission test: only compute-bound, high-FLOP-per-byte,
       residency-reused kernels belong. Memory-bound relational ops stay in DuckDB.
     - Control/data-plane separation: commands (query/kernel ref + params) go over NATS; bulk
       data must NOT ride the bus — the worker loads locally or via shared mmap/Arrow IPC.
     - Correctness of side effects: any publishing/messaging path is at-least-once with
       idempotent keys; UDFs may run multiple times / in parallel / on aborted queries. No
       duplicated, missing, or approximate results; GPU always has an exact CPU fallback.
     - Numerical correctness of kernels vs a CPU/BLAS reference (nulls, dtypes, overflow),
       memory safety in the Arrow/GPU handoff, and removal of kernels not proven faster.

6. Constitution: run /speckit-constitution with these principles (adapt wording, keep intent):
     I.   Validation before construction. No gateway, no kernel library, no extension is built
          until a throwaway spike proves the founding claim on this machine: a GPU-enabled
          DuckDB sidecar that receives a QUERY over NATS, loads its input LOCALLY, runs a
          resident-session compute-bound workload (PCA/power-iteration), and replies with the
          small result — beating an in-process CPU baseline by >=3x end to end. If it does not
          clear the bar, the project stops and reports the negative result.
     II.  Benchmark-gated increments. Every kernel is benchmarked end to end vs a CPU/BLAS
          reference before the next is started; conversion/handoff and the worker's local load
          are always counted; kernels that do not beat CPU are removed, not optimized
          speculatively.
     III. Arithmetic intensity is the admission criterion. Only compute-bound, high-FLOP-per-
          byte, residency-reused numerical kernels are in scope. Relational/memory-bound
          operators are explicitly out of scope — they belong to DuckDB's CPU engine. (This is
          the settled result of the prior project: relational GPU offload lost ~12x; PCA with a
          resident session won ~5x.)
     IV.  Control/data-plane separation. NATS carries only the command (a reference to the
          input + params), O(1) in data size. Bulk data never rides the bus; the worker loads
          locally (GPU-enabled DuckDB sidecar running the query) or via shared memory. Keeping
          the worker's GPU session resident across commands is mandatory — a single op per load
          is extraction-bound and does not clear the bar.
     V.   Fallback mandatory; no silent wrong answers; exactly-once intent for side effects.
          Unsupported ops/types fall back to exact CPU execution. Messaging/publishing paths are
          at-least-once with idempotent keys and must not emit irreversibly before commit. No
          approximate or partial result without an explicit flag.
     VI.  Portability is a decision, not a default. Apple-only (MLX/Metal) vs portable
          (Vulkan/SPIR-V, e.g. Cyfra via MoltenVK) is an explicit, recorded architecture
          decision justified against the goal, with rejected options and reasons — not assumed.
          Default unless portability is a stated primary goal: MLX in-process for the spike,
          then measure the out-of-process sidecar (NATS command + local load) before choosing
          a portable/JVM path.

7. README: a short skeleton stating what Pinion is (one numerical-compute layer on top of
   DuckDB, not a relational accelerator), the validation-gated approach, and a Status line
   noting the founding spike has not yet run.

8. Commit everything on the branch with a clear message. Do not push unless I ask.

Reference context (from the prior project, already validated — do not re-derive):
- Relational GPU offload on Apple Silicon LOSES (memory-bound; ~12x slower, worse with scale).
- Compute-bound numerical offload WINS ~5x, but only load-once-compute-many with a resident
  GPU session; a single op per load is ~1.3x (extraction-bound).
- On Apple/MLX 0.32: no int64 GPU scatter, no boolean-mask indexing, float64 downcasts to
  float32 (no doubles) — shape kernels accordingly (exact int via int64 cumsum, float via
  per-group scatter with documented tolerance).
- DuckDB has Arrow-vectorized UDFs (`create_function(..., type="arrow")`) and a read-only NATS
  scan extension (`nats_js`) already exists — reuse it for inbound; build only outbound.
```

---

## After init

Once bootstrapped, the natural first move is the founding spike (Constitution Principle I):
a `pinion-worker` sidecar (GPU-enabled DuckDB + MLX) that subscribes to a NATS subject, receives
a query + kernel command, runs the query locally to materialize input, executes the resident-
session PCA/power-iteration, and replies with the small result — benchmarked end to end against
the in-process baseline from this project's `spike/bench_pca.py`. If it holds ~5×, Pinion is
worth building; if the sidecar overhead erodes it below 3×, that is a documented stop.

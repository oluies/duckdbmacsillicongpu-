# Design note: a NATS UDF gateway for DuckDB (command / streaming / CDC)

**Status**: exploratory. Idea: a Python/C++ DuckDB UDF that talks to external processes over
[NATS](https://nats.io), for three patterns — command (request-reply), streaming egress, and
change capture. Generalizes the GPU gateway ([`udf-gateway.md`](./udf-gateway.md)): NATS is a
clean transport to an *out-of-process* compute service, GPU or otherwise.

## What already exists — don't rebuild the read path

[`nats_js`](https://duckdb.org/community_extensions/extensions/nats_js)
([brannn/duckdb-nats-jetstream](https://github.com/brannn/duckdb-nats-jetstream), C++, MIT) is a
**read-only** extension: `nats_scan()` consumes JetStream messages into tables, with
timestamp/subject/sequence filtering and JSON/protobuf field extraction. It does **not**
publish, request-reply, do command semantics, or CDC. So the inbound "NATS as a table source"
side is solved; the gap is everything **outbound / bidirectional**.

## The three patterns

### 1. Command pattern — request-reply (the highest-value one)

An Arrow-vectorized UDF publishes a request to a NATS subject and awaits a reply — a generic
RPC from SQL to any external worker. NATS core has native request-reply, so this is a natural
fit.

```sql
SELECT id, gpu_pca(embedding, 20) FROM vectors;   -- UDF -> NATS req -> worker -> reply
```

This is the **transport for the out-of-process GPU kernel service** in `udf-gateway.md`
(Option B / Cyfra, or a remote MLX box), and it decouples any external enrichment, model
inference, or legacy service call.

### Split the control plane from the data plane

The important design choice: **NATS carries only the command, not the bulk data.** The request
is small and O(1) in data size — a kernel id + a *reference* to the input (a SQL query, a
table/file path, a sequence range, a shared-memory handle) + params. The worker pulls the
actual columns **out of band**, by whatever path is fastest:

- **On one box (the common case)**: the worker is a GPU-enabled DuckDB sidecar that opens the
  same database/Parquet read-only and runs the query itself, materializing its input via
  DuckDB's own local Arrow export — so there is **no cross-process Arrow serialization at all**.
  Or a shared `mmap` / Arrow IPC file both sides agree on, near zero-copy on unified memory.
- **Across machines**: shared object storage (S3), or Arrow Flight for the bulk stream, with
  NATS still only signaling "run kernel K over query Q, params P".

The reply over NATS is likewise small — the result itself when it is tiny (PCA eigenvalues,
a scalar, a class label), or a *pointer* to where a large result was written (shared file /
object store), never the large result inlined.

**This dissolves most of the round-trip caveat.** The concern was that a NATS hop +
serialization would compete with the data transfer and erode the win. With control/data split,
the NATS message is just a control signal (sub-millisecond, data-size-independent); the only
cost that remains is the *worker's own* data load — which is the same load-once-compute-many
residency question as the in-process gateway, now cleanly separated:

- **Keep the worker's session resident.** The matrix must live in the worker across commands (a
  handle model: `load(ref) -> handle`, then `pca(handle, k)`), or it re-pays extraction every
  command. But note that extraction is now a *local* DuckDB→GPU load in the worker, not a
  cross-process transfer — the fast path the PCA spike already measured at ~5×.
- **Batch at the command level, not per row.** One command processes a whole query/partition,
  not one row.

The upshot: the earlier "handoff is the bottleneck" lesson still says *keep data resident and
load locally* — but moving the command over NATS costs essentially nothing, because the data
never rides the bus.

### 2. Streaming egress — DuckDB as a producer

A UDF (or a sink pattern: `INSERT INTO ... SELECT` over a publishing UDF) emits rows/events to
NATS subjects, feeding downstream consumers. JetStream gives persistence + at-least-once.
Straightforward and useful for wiring query output into an event mesh.

### 3. CDC — honest scope

This is the one to be careful about. **DuckDB is not a log-based CDC source.** It has no
triggers and no logical-decoding / WAL-tail API like Postgres, so there is no clean hook to emit
a change event exactly when a row changes. What is actually achievable:

- **Watermark polling**: a scheduled query over a monotonic column (sequence / updated_at) that
  publishes the new/changed rows since the last watermark. This is "detect changes via SQL and
  publish," not true log-based CDC — it misses in-place updates without a version column and
  can't see deletes without tombstones.
- **Write-path wrapping**: route all writes through a UDF/procedure that also publishes. Couples
  ingestion to NATS and only covers writes that go through that path.

Neither is the Debezium-style guarantee people mean by "CDC." Worth prototyping the watermark
variant, but label it plainly so it isn't mistaken for log-based capture.

## The correctness landmine (constitution Principle IV applies)

A UDF with side effects (publishing) is dangerous inside a query engine that may **call UDFs
multiple times, in parallel, out of order, or on a query that later aborts**. That risks
duplicate or missing emits — the messaging analogue of "silent wrong answers." Any publishing
UDF must therefore:

- Treat delivery as **at-least-once** and make consumers **idempotent** (carry a stable
  message key: row id + query/txn id + subject).
- Not assume it runs exactly once per row, and not emit irreversibly before the query commits
  (buffer + publish on success where the pattern allows).
- Offer a CPU/no-op fallback and never silently drop messages on broker unavailability — surface
  the error.

## Why NATS specifically

- Native **request-reply** (command pattern) in core NATS; **JetStream** adds persistence and
  at-least-once for streaming/CDC — one system covers all three patterns.
- Lightweight, language-agnostic clients (Python, C++, Scala/JVM) — so the same bus can reach an
  MLX worker, a Cyfra JVM kernel service, or any external process.
- Decouples DuckDB from the compute/runtime, which is exactly what makes the out-of-process GPU
  option viable without the in-process JVM boundary the constitution flagged.

## Recommendation

- **Reuse `nats_js` for inbound.** Build only the outbound/bidirectional side.
- **Start with the request-reply command UDF**, control/data split (command over NATS, data
  loaded locally by the worker) — this unlocks the portable (Cyfra) or remote-MLX story from
  `udf-gateway.md`. Spike it the honest way: a GPU-enabled DuckDB sidecar that receives a query
  over NATS, loads its input locally, runs the resident-session PCA workload, and replies with
  the small result. Measure end to end against the in-process baseline; the question is now
  whether the worker's *local* load-once-compute-many holds the ~5× (it should — same fast path
  the spike measured), plus a sub-millisecond control hop — not whether a bulk transfer over the
  bus survives.
- **Streaming egress** second (low risk).
- **CDC**: prototype watermark-polling only, labeled as not log-based.

All three would be a separate project from the relational accelerator, validated on its own.

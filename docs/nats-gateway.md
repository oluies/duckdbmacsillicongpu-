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
(Option B / Cyfra, or a remote MLX box): DuckDB (C++) never links the compute runtime — it
sends an Arrow batch over NATS and gets Arrow back. It also decouples any external enrichment,
model inference, or legacy service call.

**But the measured lesson carries over, harder.** The PCA spike showed the *handoff* is the
bottleneck, not the compute. A NATS round-trip adds a process hop + serialization on top. So:

- **Batch, never per-row.** Use the Arrow-vectorized UDF (`create_function(..., type="arrow")`,
  verified working) so one request carries a whole column chunk, not one row. Per-row
  request-reply would be latency-bound and pointless.
- **Keep sessions resident.** For the GPU case the matrix must live in the worker across calls
  (a handle model), or you re-pay extraction *and* the NATS round-trip every op — back under
  the 3× bar.

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
- **Start with the request-reply command UDF**, Arrow-batched, as the transport for the
  out-of-process GPU gateway — this is the piece that unlocks the portable (Cyfra) or
  remote-MLX story from `udf-gateway.md`. Spike it the same honest way: measure the Arrow +
  NATS round-trip against an in-process baseline on the PCA workload; if the round-trip erodes
  the ~5× resident-session win below 3×, that scopes where an external gateway is worth it.
- **Streaming egress** second (low risk).
- **CDC**: prototype watermark-polling only, labeled as not log-based.

All three would be a separate project from the relational accelerator, validated on its own.

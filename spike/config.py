"""Fixed constants for the Phase 0 spike (task T004).

Everything reproducible lives here so the CPU-DuckDB run and the Apple-GPU run
read byte-identical input. See specs/001-duckdb-gpu-accelerator/data-model.md.
"""

SEED = 42

# A few hundred million rows (spec FR-001). Tunable via CLI; this is the gate size.
ROWS = 300_000_000

# Moderate group-key cardinality: neither trivially few nor near-unique.
CARDINALITY = 100_000

# NULL key is its own group (DuckDB semantics). On the GPU we cannot carry a
# validity bitmap through argsort, so NULL keys are mapped to this sentinel,
# which is deliberately outside [0, CARDINALITY). DuckDB's NULL-key group is
# mapped to the same sentinel before comparison so the two paths line up.
NULL_KEY_SENTINEL = CARDINALITY  # 100_000, guaranteed not a real key

# Roughly 1-in-N rows get a NULL key / NULL val_d. val_i is kept non-null so the
# exact integer-sum correctness signal stays crisp (a group is never all-NULL,
# which would make DuckDB SUM return NULL while a fill-with-0 GPU sum returns 0).
NULL_KEY_EVERY = 1000  # ~0.1% NULL keys -> exercises the null-group path
NULL_VALD_EVERY = 20   # ~5%   NULL val_d -> exercises SUM-skips-null on the tolerance column

DEFAULT_DB = "data/bench.duckdb"

# Phase 0 continuation gate (constitution / spec FR-007): GPU must beat CPU
# DuckDB by at least this ratio, end to end, on a verified-correct result.
GATE_RATIO = 3.0

# Relative tolerance for the floating SUM only. Integer SUM and COUNT are exact.
# The double column is accumulated in float32 (Apple GPUs have no float64) in a
# per-group scatter, so its sum differs from DuckDB's order by a small relative
# amount. See research.md R6.
FLOAT_SUM_RTOL = 1e-3

"""Honest-timing helper for the spike (task T005).

The whole point of Phase 0 is that the GPU number is not a lie. The GPU timing
span MUST include Arrow export + conversion to MLX + compute + mx.eval() (MLX is
lazy) + materialization back to host. This helper just wraps perf_counter; the
caller is responsible for putting *all* of that inside the `with` block. See
contract G1 in specs/001-duckdb-gpu-accelerator/contracts/spike-cli.md and
research.md R5.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(store: list):
    """Append the wall-clock seconds of the block to `store`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store.append(time.perf_counter() - t0)


def best_of(times: list) -> float:
    """Best (minimum) wall-clock over the measured runs."""
    return min(times)

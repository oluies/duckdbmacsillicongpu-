"""Numerical-equality check: GPU result vs DuckDB CPU result (task T009).

Integer SUM and COUNT must match DuckDB exactly (bit-for-bit). The floating SUM
is compared within a documented relative tolerance, because the double column is
accumulated in float32 on the GPU (Apple GPUs have no float64) in a different
order than DuckDB's CPU sum. See research.md R6 and spec FR-004 / SC-003.

Each result is a dict: {key(int) -> (sum_i(int), sum_d(float), count(int))},
with the NULL-key group represented by config.NULL_KEY_SENTINEL on both sides.
"""

from __future__ import annotations

import math

import config


def verify(cpu: dict, gpu: dict, rtol: float = config.FLOAT_SUM_RTOL) -> tuple[bool, str]:
    if cpu.keys() != gpu.keys():
        only_cpu = set(cpu) - set(gpu)
        only_gpu = set(gpu) - set(cpu)
        return False, (
            f"group-key sets differ: {len(cpu)} CPU vs {len(gpu)} GPU groups; "
            f"{len(only_cpu)} only in CPU, {len(only_gpu)} only in GPU"
        )

    worst_rel = 0.0
    for k in cpu:
        c_si, c_sd, c_c = cpu[k]
        g_si, g_sd, g_c = gpu[k]
        if int(c_si) != int(g_si):
            return False, f"integer SUM mismatch for key {k}: CPU {c_si} != GPU {g_si}"
        if int(c_c) != int(g_c):
            return False, f"COUNT mismatch for key {k}: CPU {c_c} != GPU {g_c}"
        # NaN/inf must be a hard mismatch, checked BEFORE the tolerance test.
        # An all-NULL val_d group makes DuckDB's SUM return NULL -> NaN here, while
        # the GPU path fills NULLs with 0.0 -> 0.0. Without this guard `rel` is NaN,
        # `NaN > rtol` is False, and the disagreement would pass silently — defeating
        # the gate's only correctness safeguard.
        if math.isnan(c_sd) != math.isnan(g_sd) or math.isinf(c_sd) or math.isinf(g_sd):
            return False, (
                f"float SUM NULL/non-finite disagreement for key {k}: "
                f"CPU {c_sd} vs GPU {g_sd} (one is NaN/inf, the other is not)"
            )
        if math.isnan(c_sd) and math.isnan(g_sd):
            continue  # both NULL for this group -> agree, nothing to tolerance-check
        denom = abs(c_sd) if abs(c_sd) > 1e-9 else 1.0
        rel = abs(c_sd - g_sd) / denom
        worst_rel = max(worst_rel, rel)
        if rel > rtol and not math.isclose(c_sd, g_sd, rel_tol=rtol, abs_tol=1e-6):
            return False, (
                f"float SUM out of tolerance for key {k}: CPU {c_sd} vs GPU {g_sd} "
                f"(rel {rel:.2e} > rtol {rtol:.0e})"
            )

    return True, (
        f"all {len(cpu):,} groups match "
        f"(int SUM + COUNT exact; float SUM worst rel err {worst_rel:.2e} <= {rtol:.0e})"
    )

"""Metric computation over simulation results."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from lancepilot.simulator import SimResult


def summarize(result: SimResult) -> dict[str, Any]:
    lat = result.latencies_ms
    return {
        "policy": result.policy,
        "queries": int(lat.size),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()),
        "hit_ratio": float(result.hits.mean()),
        "remote_requests": result.remote_requests,
        "remote_bytes": result.remote_bytes,
        "evictions": result.evictions,
        "max_used_bytes": result.max_used_bytes,
        "capacity_bytes": result.capacity_bytes,
    }


def rolling_hit_ratio(hits: npt.NDArray[np.bool_], window: int) -> npt.NDArray[np.float64]:
    """Trailing-window hit ratio; element i covers queries max(0, i-window+1)..i."""
    c = np.concatenate([[0.0], np.cumsum(hits.astype(np.float64))])
    idx = np.arange(1, hits.size + 1)
    start = np.maximum(idx - window, 0)
    out: npt.NDArray[np.float64] = (c[idx] - c[start]) / (idx - start)
    return out


def phase_shift_metrics(
    hits: npt.NDArray[np.bool_],
    shift_at: int,
    window: int,
    steady_window: int,
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Pre-shift steady state = hit ratio over the `steady_window` queries before the shift.
    Post-shift steady state = hit ratio over the final `steady_window` queries.

    Recovery: the smallest number of post-shift queries r >= window such that the
    trailing `window`-query hit ratio, computed only over post-shift queries, reaches
    threshold * pre-shift steady state. None if it never recovers."""
    pre = float(hits[max(0, shift_at - steady_window) : shift_at].mean())
    post = float(hits[-steady_window:].mean())
    target = threshold * pre
    post_hits = hits[shift_at:]
    roll = rolling_hit_ratio(post_hits, window)
    recovered = np.nonzero((np.arange(post_hits.size) >= window - 1) & (roll >= target))[0]
    first_window = float(post_hits[:window].mean())
    return {
        "pre_shift_hit_ratio": pre,
        "post_shift_steady_hit_ratio": post,
        "first_window_after_shift_hit_ratio": first_window,
        "recovery_target_hit_ratio": target,
        "recovery_queries": int(recovered[0]) + 1 if recovered.size else None,
        "rolling_window": window,
    }

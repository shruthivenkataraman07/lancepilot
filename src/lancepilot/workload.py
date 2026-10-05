"""Deterministic synthetic workload generators. All data produced here is synthetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lancepilot.models import CacheObject, QueryRequest

KB = 1024
MB = 1024 * 1024


@dataclass(frozen=True)
class ObjectSpec:
    num_objects: int = 1000
    size_median_kb: float = 256.0
    size_sigma: float = 0.6
    min_size_kb: float = 4.0
    max_size_kb: float = 16 * 1024.0
    num_partitions: int = 8
    partition_latency_ms: tuple[float, float] = (15.0, 80.0)
    bandwidth_mb_s: float = 100.0
    jitter_sigma: float = 0.2


@dataclass(frozen=True)
class Workload:
    """A synthetic object catalog plus a query trace over it."""

    objects: list[CacheObject]
    trace: list[QueryRequest]
    meta: dict[str, Any] = field(default_factory=dict)


def generate_objects(spec: ObjectSpec, rng: np.random.Generator) -> list[CacheObject]:
    """Create objects with lognormal sizes and partition-dependent remote fetch cost.

    remote_fetch_ms = partition first-byte latency * per-object jitter + size / bandwidth
    """
    sizes_kb = rng.lognormal(np.log(spec.size_median_kb), spec.size_sigma, spec.num_objects)
    sizes = np.clip(sizes_kb, spec.min_size_kb, spec.max_size_kb) * KB
    lo, hi = spec.partition_latency_ms
    partition_latency = rng.uniform(lo, hi, spec.num_partitions)
    partitions = rng.integers(0, spec.num_partitions, spec.num_objects)
    jitter = rng.lognormal(0.0, spec.jitter_sigma, spec.num_objects)
    transfer_ms = sizes / (spec.bandwidth_mb_s * MB) * 1000.0
    fetch_ms = partition_latency[partitions] * jitter + transfer_ms
    return [
        CacheObject(
            object_id=i,
            size_bytes=int(sizes[i]),
            remote_fetch_ms=round(float(fetch_ms[i]), 4),
            partition_id=int(partitions[i]),
        )
        for i in range(spec.num_objects)
    ]


def _to_trace(object_ids: np.ndarray) -> list[QueryRequest]:
    return [
        QueryRequest(query_id=i, required_object_id=int(oid), timestamp=i)
        for i, oid in enumerate(object_ids)
    ]


def phase_shift_trace(
    num_objects: int,
    num_queries: int,
    shift_at: int,
    hot_a: tuple[int, int],
    hot_b: tuple[int, int],
    hot_fraction: float,
    rng: np.random.Generator,
) -> list[QueryRequest]:
    """Before `shift_at`, `hot_fraction` of queries go uniformly to hot range A; afterwards
    to hot range B. Remaining queries go uniformly to objects outside the current hot range."""
    ids = np.empty(num_queries, dtype=np.int64)
    all_ids = np.arange(num_objects)
    for start, stop, (h0, h1) in ((0, shift_at, hot_a), (shift_at, num_queries, hot_b)):
        n = stop - start
        hot = np.arange(h0, h1)
        cold = all_ids[(all_ids < h0) | (all_ids >= h1)]
        is_hot = rng.random(n) < hot_fraction
        ids[start:stop] = np.where(is_hot, rng.choice(hot, n), rng.choice(cold, n))
    return _to_trace(ids)


def zipf_trace(
    num_objects: int, num_queries: int, zipf_s: float, rng: np.random.Generator
) -> list[QueryRequest]:
    """Zipf(s) popularity over a random permutation of objects, so rank is independent of
    size and remote cost."""
    ranks = np.arange(1, num_objects + 1, dtype=np.float64)
    probs = ranks**-zipf_s
    probs /= probs.sum()
    order = rng.permutation(num_objects)
    ids = order[rng.choice(num_objects, num_queries, p=probs)]
    return _to_trace(ids)


def build_workload(workload_cfg: dict[str, Any], object_cfg: dict[str, Any]) -> Workload:
    """Build a workload from config dicts. One seed drives objects and trace."""
    seed = int(workload_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    num_objects = int(workload_cfg.get("num_objects", 1000))
    obj_kwargs = dict(object_cfg)
    if "partition_latency_ms" in obj_kwargs:
        obj_kwargs["partition_latency_ms"] = tuple(obj_kwargs["partition_latency_ms"])
    spec = ObjectSpec(num_objects=num_objects, **obj_kwargs)
    objects = generate_objects(spec, rng)
    num_queries = int(workload_cfg.get("num_queries", 10_000))
    kind = workload_cfg["type"]
    meta: dict[str, Any] = {"type": kind, "seed": seed, "synthetic": True}
    if kind == "phase_shift":
        shift_at = int(workload_cfg.get("shift_at", num_queries // 2))
        hot_a = tuple(workload_cfg.get("hot_a", (0, 50)))
        hot_b = tuple(workload_cfg.get("hot_b", (50, 100)))
        trace = phase_shift_trace(
            num_objects,
            num_queries,
            shift_at,
            (int(hot_a[0]), int(hot_a[1])),
            (int(hot_b[0]), int(hot_b[1])),
            float(workload_cfg.get("hot_fraction", 0.8)),
            rng,
        )
        meta["shift_at"] = shift_at
    elif kind == "zipf":
        trace = zipf_trace(num_objects, num_queries, float(workload_cfg.get("zipf_s", 0.9)), rng)
    else:
        raise ValueError(f"unknown workload type: {kind}")
    return Workload(objects=objects, trace=trace, meta=meta)

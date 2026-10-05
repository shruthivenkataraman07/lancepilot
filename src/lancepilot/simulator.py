"""Trace-driven simulator: RAM cache in front of a simulated remote object store.
No real sleeping; latency is accounted, not waited for."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lancepilot.models import CacheObject, QueryRequest
from lancepilot.policies import CachePolicy


class SimulatedRemoteStore:
    """Counts remote requests and bytes; returns each object's simulated fetch latency."""

    def __init__(self, objects: Sequence[CacheObject]) -> None:
        self._objects = {o.object_id: o for o in objects}
        self.requests = 0
        self.bytes_read = 0

    def get(self, object_id: int) -> CacheObject:
        return self._objects[object_id]

    def fetch(self, object_id: int) -> float:
        obj = self._objects[object_id]
        self.requests += 1
        self.bytes_read += obj.size_bytes
        return obj.remote_fetch_ms


@dataclass
class SimResult:
    policy: str
    latencies_ms: npt.NDArray[np.float64]
    hits: npt.NDArray[np.bool_]
    remote_requests: int
    remote_bytes: int
    evictions: int
    max_used_bytes: int
    capacity_bytes: int


def simulate(
    policy: CachePolicy,
    objects: Sequence[CacheObject],
    trace: Sequence[QueryRequest],
    hit_latency_ms: float = 0.1,
) -> SimResult:
    store = SimulatedRemoteStore(objects)
    n = len(trace)
    latencies = np.empty(n, dtype=np.float64)
    hits = np.zeros(n, dtype=np.bool_)
    max_used = 0
    for i, q in enumerate(trace):
        obj = store.get(q.required_object_id)
        if policy.lookup(obj, q.timestamp):
            hits[i] = True
            latencies[i] = hit_latency_ms
        else:
            latencies[i] = store.fetch(obj.object_id)
            policy.on_miss(obj, q.timestamp)
        used = policy.cache.used_bytes
        if used > policy.cache.capacity_bytes:
            raise AssertionError("capacity invariant violated")
        max_used = max(max_used, used)
    return SimResult(
        policy=policy.name,
        latencies_ms=latencies,
        hits=hits,
        remote_requests=store.requests,
        remote_bytes=store.bytes_read,
        evictions=policy.cache.evictions,
        max_used_bytes=max_used,
        capacity_bytes=policy.cache.capacity_bytes,
    )

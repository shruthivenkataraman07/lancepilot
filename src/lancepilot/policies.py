"""Cache policies: LRU baseline and the LancePilot workload-aware policy."""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from lancepilot.cache import RamCache
from lancepilot.models import CacheObject


class CachePolicy(ABC):
    """A policy owns a RamCache. The simulator calls `lookup` for every query and
    `on_miss` after a remote fetch; `on_miss` returns whether the object was admitted."""

    name: str = "policy"

    def __init__(self, capacity_bytes: int) -> None:
        self.cache = RamCache(capacity_bytes)

    @abstractmethod
    def lookup(self, obj: CacheObject, now: int) -> bool: ...

    @abstractmethod
    def on_miss(self, obj: CacheObject, now: int) -> bool: ...


class LRUPolicy(CachePolicy):
    """Admit every miss that can fit; evict least recently used until it fits."""

    name = "LRU"

    def __init__(self, capacity_bytes: int) -> None:
        super().__init__(capacity_bytes)
        self._order: OrderedDict[int, None] = OrderedDict()

    def lookup(self, obj: CacheObject, now: int) -> bool:
        if obj.object_id in self._order:
            self._order.move_to_end(obj.object_id)
            return True
        return False

    def on_miss(self, obj: CacheObject, now: int) -> bool:
        if not self.cache.can_ever_fit(obj.size_bytes):
            return False
        while self.cache.free_bytes < obj.size_bytes:
            victim, _ = self._order.popitem(last=False)
            self.cache.evict(victim)
        self.cache.insert(obj.object_id, obj.size_bytes)
        self._order[obj.object_id] = None
        return True


class LancePilotPolicy(CachePolicy):
    """Workload-aware admission and eviction.

    heat_i(t)    = sum over past accesses k of (1 - alpha) ** (t - t_k)
                   (equivalently: heat <- (1 - alpha) ** dt * heat + 1 on each access)
    size_weight  = max(size_bytes / median_object_size, epsilon)
    value_i      = heat_i * remote_fetch_ms_i / size_weight_i

    Because every heat decays by the same factor per query, the ordering of values at any
    instant is unchanged by decay. We therefore store the time-invariant log score
        S_i = log(sum_k (1 - alpha) ** -t_k) + log(remote_fetch_ms) - log(size_weight)
    which only changes when object i is accessed. This lets residents sit in a heap.

    Admission: a miss is admitted if it fits in free space; otherwise the lowest-value
    residents (ties broken by object_id) are selected until enough bytes would be free,
    and the miss is admitted only if its value is strictly greater than every selected
    victim's value. If not, nothing is evicted and the object bypasses the cache.
    """

    name = "LancePilot"

    def __init__(
        self,
        capacity_bytes: int,
        median_object_size: float,
        alpha: float = 0.01,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__(capacity_bytes)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.median_object_size = float(median_object_size)
        self.alpha = alpha
        self.epsilon = epsilon
        self._log_decay = math.log1p(-alpha)
        self._log_heat: dict[int, float] = {}  # time-invariant log heat, all seen objects
        self._resident_score: dict[int, float] = {}
        self._heap: list[tuple[float, int]] = []  # lazy min-heap of (score, object_id)
        self.rejected_admissions = 0

    # -- scoring -------------------------------------------------------------------------
    def _record_access(self, obj: CacheObject, now: int) -> None:
        contribution = -now * self._log_decay  # log((1 - alpha) ** -now)
        prev = self._log_heat.get(obj.object_id)
        self._log_heat[obj.object_id] = (
            contribution if prev is None else float(np.logaddexp(prev, contribution))
        )

    def score(self, obj: CacheObject) -> float:
        """Time-invariant log value; higher means more worth caching."""
        size_weight = max(obj.size_bytes / self.median_object_size, self.epsilon)
        return (
            self._log_heat.get(obj.object_id, -math.inf)
            + math.log(max(obj.remote_fetch_ms, 1e-9))
            - math.log(size_weight)
        )

    def heat(self, object_id: int, now: int) -> float:
        """Actual decayed heat at logical time `now` (for inspection and tests)."""
        log_h = self._log_heat.get(object_id)
        return 0.0 if log_h is None else math.exp(log_h + now * self._log_decay)

    def _set_resident_score(self, object_id: int, s: float) -> None:
        self._resident_score[object_id] = s
        heapq.heappush(self._heap, (s, object_id))

    def _pop_lowest(self) -> tuple[float, int]:
        while True:
            s, oid = heapq.heappop(self._heap)
            if self._resident_score.get(oid) == s:
                return s, oid

    # -- policy interface ----------------------------------------------------------------
    def lookup(self, obj: CacheObject, now: int) -> bool:
        self._record_access(obj, now)
        if obj.object_id in self._resident_score:
            self._set_resident_score(obj.object_id, self.score(obj))
            return True
        return False

    def on_miss(self, obj: CacheObject, now: int) -> bool:
        if not self.cache.can_ever_fit(obj.size_bytes):
            self.rejected_admissions += 1
            return False
        candidate = self.score(obj)
        victims: list[tuple[float, int]] = []
        freed = self.cache.free_bytes
        while freed < obj.size_bytes:
            s, oid = self._pop_lowest()
            victims.append((s, oid))
            freed += self.cache.size_of(oid)
        if victims and candidate <= max(s for s, _ in victims):
            for entry in victims:  # roll back: victims stay resident
                heapq.heappush(self._heap, entry)
            self.rejected_admissions += 1
            return False
        for _, oid in victims:
            del self._resident_score[oid]
            self.cache.evict(oid)
        self.cache.insert(obj.object_id, obj.size_bytes)
        self._set_resident_score(obj.object_id, candidate)
        if len(self._heap) > 4 * len(self._resident_score) + 1024:
            self._heap = [(s, o) for o, s in self._resident_score.items()]
            heapq.heapify(self._heap)
        return True



import numpy as np
import pytest
from conftest import obj

from lancepilot.cache import CapacityError, RamCache
from lancepilot.policies import CachePolicy, LancePilotPolicy, LRUPolicy
from lancepilot.simulator import simulate
from lancepilot.workload import ObjectSpec, generate_objects, zipf_trace


def _policies(capacity: int, median: float) -> list[CachePolicy]:
    return [LRUPolicy(capacity), LancePilotPolicy(capacity, median_object_size=median)]


def test_ram_cache_rejects_overflowing_insert() -> None:
    c = RamCache(100)
    c.insert(1, 60)
    with pytest.raises(CapacityError):
        c.insert(2, 41)
    assert c.used_bytes == 60


@pytest.mark.parametrize("which", [0, 1])
def test_oversized_object_rejected(which: int) -> None:
    p = _policies(100, 50.0)[which]
    small, huge = obj(1, 50), obj(2, 101, cost=1e6)
    p.lookup(small, 0)
    p.on_miss(small, 0)
    for t in range(1, 20):
        assert not p.lookup(huge, t)
        assert not p.on_miss(huge, t)
    assert 2 not in p.cache and 1 in p.cache
    assert p.cache.used_bytes == 50


@pytest.mark.parametrize("which", [0, 1])
def test_capacity_never_exceeded_on_random_trace(which: int) -> None:
    rng = np.random.default_rng(0)
    objects = generate_objects(ObjectSpec(num_objects=300, size_sigma=1.5), rng)
    trace = zipf_trace(300, 5000, 0.8, rng)
    capacity = sum(o.size_bytes for o in objects) // 20
    median = float(np.median([o.size_bytes for o in objects]))
    p = _policies(capacity, median)[which]
    res = simulate(p, objects, trace)  # simulate asserts the invariant after every query
    assert res.max_used_bytes <= capacity
    assert p.cache.used_bytes == sum(p.cache.size_of(o.object_id) for o in objects
                                     if o.object_id in p.cache)

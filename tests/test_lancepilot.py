from conftest import obj

from lancepilot.policies import LancePilotPolicy


def test_prefers_expensive_object_of_same_size() -> None:
    p = LancePilotPolicy(capacity_bytes=100, median_object_size=100, alpha=0.01)
    cheap, costly = obj(1, 100, cost=5.0), obj(2, 100, cost=80.0)
    p.lookup(cheap, 0)
    assert p.on_miss(cheap, 0)
    p.lookup(costly, 1)
    assert p.on_miss(costly, 1)  # higher value displaces lower value
    assert 2 in p.cache and 1 not in p.cache


def test_rejects_lower_value_candidate() -> None:
    p = LancePilotPolicy(capacity_bytes=100, median_object_size=100, alpha=0.01)
    costly, cheap = obj(1, 100, cost=80.0), obj(2, 100, cost=5.0)
    p.lookup(costly, 0)
    p.on_miss(costly, 0)
    p.lookup(cheap, 1)
    assert not p.on_miss(cheap, 1)
    assert 1 in p.cache and 2 not in p.cache
    assert p.cache.evictions == 0 and p.rejected_admissions == 1


def test_prefers_small_object_at_equal_cost_and_heat() -> None:
    p = LancePilotPolicy(capacity_bytes=100, median_object_size=50, alpha=0.01)
    big, small = obj(1, 100, cost=20.0), obj(2, 40, cost=20.0)
    p.lookup(big, 0)
    p.on_miss(big, 0)
    p.lookup(small, 1)
    assert p.on_miss(small, 1)
    assert 2 in p.cache and 1 not in p.cache


def test_heat_decays_and_accumulates() -> None:
    p = LancePilotPolicy(capacity_bytes=1000, median_object_size=100, alpha=0.1)
    o = obj(1, 100)
    p.lookup(o, 0)
    assert abs(p.heat(1, 0) - 1.0) < 1e-12
    assert abs(p.heat(1, 10) - 0.9**10) < 1e-12
    p.lookup(o, 10)
    assert abs(p.heat(1, 10) - (1.0 + 0.9**10)) < 1e-12


def test_frequently_accessed_object_survives_scan() -> None:
    """A hot object should not be flushed by a stream of one-hit cold objects."""
    p = LancePilotPolicy(capacity_bytes=200, median_object_size=100, alpha=0.01)
    hot = obj(0, 100, cost=20.0)
    t = 0
    for _ in range(10):
        if not p.lookup(hot, t):
            p.on_miss(hot, t)
        t += 1
    for oid in range(1, 50):
        cold = obj(oid, 100, cost=20.0)
        if not p.lookup(cold, t):
            p.on_miss(cold, t)
        t += 1
    assert 0 in p.cache

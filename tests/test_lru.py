from conftest import obj

from lancepilot.policies import LRUPolicy


def test_lru_evicts_least_recently_used() -> None:
    p = LRUPolicy(capacity_bytes=300)
    a, b, c, d = obj(1, 100), obj(2, 100), obj(3, 100), obj(4, 100)
    for t, o in enumerate((a, b, c)):
        assert not p.lookup(o, t)
        assert p.on_miss(o, t)
    assert p.lookup(a, 3)  # a becomes most recent; b is now LRU
    assert not p.lookup(d, 4)
    p.on_miss(d, 4)
    assert 2 not in p.cache
    assert all(k in p.cache for k in (1, 3, 4))
    assert p.cache.evictions == 1


def test_lru_evicts_multiple_for_large_object() -> None:
    p = LRUPolicy(capacity_bytes=300)
    for t, o in enumerate((obj(1, 100), obj(2, 100), obj(3, 100))):
        p.lookup(o, t)
        p.on_miss(o, t)
    big = obj(9, 250)
    p.lookup(big, 3)
    assert p.on_miss(big, 3)
    assert 9 in p.cache and len(p.cache) == 1
    assert p.cache.evictions == 3

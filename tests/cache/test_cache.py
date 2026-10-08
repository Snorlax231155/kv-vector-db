from __future__ import annotations

import asyncio

import pytest

from medmemory.cache import LRUCache, SingleFlight, exact_key


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_lru_evicts_least_recently_used() -> None:
    c: LRUCache[int] = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # a is now most recent
    c.set("c", 3)  # evicts b
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3
    s = c.stats()
    assert s.evictions == 1 and s.hits == 3 and s.misses == 1


def test_ttl_expiry_counts_as_miss_and_expiration() -> None:
    clock = Clock()
    c: LRUCache[str] = LRUCache(10, default_ttl_s=5, clock=clock)
    c.set("k", "v")
    clock.t = 4.9
    assert c.get("k") == "v"
    clock.t = 5.1
    assert c.get("k") is None
    assert c.stats().expirations == 1


def test_per_entry_ttl_overrides_default() -> None:
    clock = Clock()
    c: LRUCache[str] = LRUCache(10, default_ttl_s=5, clock=clock)
    c.set("long", "v", ttl_s=100)
    clock.t = 50
    assert c.get("long") == "v"


def test_tag_invalidation_is_write_through() -> None:
    c: LRUCache[str] = LRUCache(10)
    c.set("q1", "a", tags=["patient:P0001"])
    c.set("q2", "b", tags=["patient:P0001", "global"])
    c.set("q3", "c", tags=["patient:P0002"])
    assert c.invalidate_tag("patient:P0001") == 2
    assert c.get("q1") is None and c.get("q2") is None and c.get("q3") == "c"
    assert c.invalidate_tag("patient:P0001") == 0


def test_overwrite_retags() -> None:
    c: LRUCache[str] = LRUCache(10)
    c.set("q", "a", tags=["patient:P0001"])
    c.set("q", "b", tags=["patient:P0002"])
    assert c.invalidate_tag("patient:P0001") == 0
    assert c.get("q") == "b"


def test_capacity_validation() -> None:
    with pytest.raises(ValueError):
        LRUCache(0)


def test_exact_key_depends_on_scope_version_and_options() -> None:
    base = exact_key("latest a1c", "P0001", None, ["patient_notes"], {"k": 5}, 1)
    assert base == exact_key("  Latest   A1C ", "P0001", None, ["patient_notes"], {"k": 5}, 1)
    assert base != exact_key("latest a1c", "P0002", None, ["patient_notes"], {"k": 5}, 1)
    assert base != exact_key("latest a1c", "P0001", None, ["patient_notes"], {"k": 5}, 2)
    assert base != exact_key("latest a1c", "P0001", None, ["patient_notes"], {"k": 3}, 1)
    assert base != exact_key(
        "latest a1c", "P0001", {"section": "x"}, ["patient_notes"], {"k": 5}, 1
    )


async def test_singleflight_coalesces_concurrent_identical_calls() -> None:
    sf: SingleFlight[int] = SingleFlight()
    calls = 0

    async def slow() -> int:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return 42

    results = await asyncio.gather(*(sf.do("same", slow) for _ in range(20)))
    assert calls == 1
    assert all(r == 42 for r, _ in results)
    assert sum(shared for _, shared in results) == 19
    assert sf.stats() == {"leaders": 1, "coalesced": 19, "in_flight": 0}


async def test_singleflight_distinct_keys_run_independently() -> None:
    sf: SingleFlight[str] = SingleFlight()

    async def make(v: str) -> str:
        await asyncio.sleep(0.01)
        return v

    out = await asyncio.gather(sf.do("a", lambda: make("a")), sf.do("b", lambda: make("b")))
    assert [r for r, _ in out] == ["a", "b"]


async def test_singleflight_propagates_errors_and_does_not_cache_them() -> None:
    sf: SingleFlight[int] = SingleFlight()

    async def boom() -> int:
        await asyncio.sleep(0.01)
        raise RuntimeError("backend down")

    results = await asyncio.gather(sf.do("k", boom), sf.do("k", boom), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)

    async def ok() -> int:
        return 1

    assert (await sf.do("k", ok))[0] == 1

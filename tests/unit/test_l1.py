"""Unit tests for L1 (in-memory LRU) tier."""

import pytest

from sagemem.tiers.l1 import L1Tier


@pytest.fixture
def l1():
    return L1Tier(capacity=3)


async def test_set_and_get(l1):
    await l1.set("k1", "hello")
    assert await l1.get("k1") == "hello"


async def test_get_missing_returns_none(l1):
    assert await l1.get("nonexistent") is None


async def test_delete(l1):
    await l1.set("k1", "v1")
    await l1.delete("k1")
    assert await l1.get("k1") is None


async def test_delete_nonexistent_is_noop(l1):
    await l1.delete("ghost")  # should not raise


async def test_exists(l1):
    assert not await l1.exists("k1")
    await l1.set("k1", 42)
    assert await l1.exists("k1")


async def test_clear(l1):
    await l1.set("k1", 1)
    await l1.set("k2", 2)
    await l1.clear()
    assert await l1.get("k1") is None
    assert len(l1) == 0


async def test_lru_eviction():
    """When capacity is exceeded, the least recently used entry is evicted."""
    l1 = L1Tier(capacity=2)
    await l1.set("a", 1)
    await l1.set("b", 2)
    await l1.set("c", 3)  # "a" should be evicted
    assert await l1.get("a") is None
    assert await l1.get("b") == 2
    assert await l1.get("c") == 3


async def test_access_updates_lru_order():
    """Accessing an entry should protect it from eviction."""
    l1 = L1Tier(capacity=2)
    await l1.set("a", 1)
    await l1.set("b", 2)
    await l1.get("a")      # "a" is now most recently used
    await l1.set("c", 3)   # "b" should be evicted, not "a"
    assert await l1.get("a") == 1
    assert await l1.get("b") is None
    assert await l1.get("c") == 3


async def test_overwrite_existing_key(l1):
    """Setting an existing key should update its value and not grow the store."""
    await l1.set("k1", "original")
    await l1.set("k1", "updated")
    assert await l1.get("k1") == "updated"
    assert len(l1) == 1


async def test_stores_various_types(l1):
    """L1 must handle any JSON-serializable Python value."""
    await l1.set("int", 42)
    await l1.set("dict", {"a": 1})
    await l1.set("list", [1, 2, 3])
    assert await l1.get("int") == 42
    assert await l1.get("dict") == {"a": 1}
    assert await l1.get("list") == [1, 2, 3]

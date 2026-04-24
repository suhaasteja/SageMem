"""Integration tests for L2 (Redis) tier. Requires a live Redis instance."""

import pytest

from sagemem.tiers.l2 import L2Tier

REDIS_URL = "redis://localhost:6379"


@pytest.fixture
async def l2():
    tier = L2Tier(url=REDIS_URL, namespace="sagemem:test:l2")
    await tier.connect()
    await tier.clear()
    yield tier
    await tier.clear()
    await tier.disconnect()


async def test_set_and_get(l2):
    await l2.set("k1", "hello")
    assert await l2.get("k1") == "hello"


async def test_get_missing_returns_none(l2):
    assert await l2.get("nonexistent") is None


async def test_delete(l2):
    await l2.set("k1", "v1")
    await l2.delete("k1")
    assert await l2.get("k1") is None


async def test_exists(l2):
    assert not await l2.exists("k1")
    await l2.set("k1", 99)
    assert await l2.exists("k1")


async def test_clear(l2):
    await l2.set("a", 1)
    await l2.set("b", 2)
    await l2.clear()
    assert await l2.get("a") is None
    assert await l2.get("b") is None


async def test_stores_dict(l2):
    payload = {"agent": "alpha", "score": 0.95}
    await l2.set("result", payload)
    assert await l2.get("result") == payload


async def test_overwrite(l2):
    await l2.set("k", "first")
    await l2.set("k", "second")
    assert await l2.get("k") == "second"

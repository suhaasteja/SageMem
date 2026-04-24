"""Integration tests for L3 (Postgres JSONB) tier. Requires a live Postgres instance."""

import pytest

from sagemem.tiers.l3 import L3Tier, VersionConflictError

PG_DSN = "postgresql://localhost/sagemem_test"


@pytest.fixture
async def l3():
    tier = L3Tier(dsn=PG_DSN, table="test_l3")
    await tier.connect()
    await tier.clear()
    yield tier
    await tier.clear()
    await tier.disconnect()


async def test_set_and_get(l3):
    await l3.set("k1", "hello")
    assert await l3.get("k1") == "hello"


async def test_get_missing_returns_none(l3):
    assert await l3.get("missing") is None


async def test_delete(l3):
    await l3.set("k1", "v")
    await l3.delete("k1")
    assert await l3.get("k1") is None


async def test_exists(l3):
    assert not await l3.exists("k1")
    await l3.set("k1", 1)
    assert await l3.exists("k1")


async def test_clear(l3):
    await l3.set("a", 1)
    await l3.set("b", 2)
    await l3.clear()
    assert await l3.get("a") is None


async def test_version_increments_on_update(l3):
    await l3.set("k", "v1")
    result = await l3.get_versioned("k")
    assert result is not None
    value, v0 = result
    await l3.set("k", "v2")
    result2 = await l3.get_versioned("k")
    assert result2 is not None
    _, v1 = result2
    assert v1 == v0 + 1


async def test_versioned_write_succeeds_on_correct_version(l3):
    await l3.set("k", "original")
    result = await l3.get_versioned("k")
    assert result is not None
    _, version = result
    await l3.set_versioned("k", "updated", expected_version=version)
    assert await l3.get("k") == "updated"


async def test_versioned_write_raises_on_stale_version(l3):
    await l3.set("k", "v1")
    await l3.set("k", "v2")  # increments version to 1
    with pytest.raises(VersionConflictError):
        await l3.set_versioned("k", "conflicted", expected_version=0)


async def test_stores_complex_value(l3):
    payload = {"tags": ["research", "memory"], "score": 0.87}
    await l3.set("doc:1", payload)
    assert await l3.get("doc:1") == payload

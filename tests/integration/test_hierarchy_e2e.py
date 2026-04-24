"""Integration e2e test for MemoryHierarchy with live tiers.

Writes to L3, reads through hierarchy, confirms L1 is populated (promoted).
Requires Redis and Postgres.
"""

import pytest

from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"


@pytest.fixture
async def hierarchy():
    l1 = L1Tier(capacity=64)
    l2 = L2Tier(url=REDIS_URL, namespace="sagemem:test:hier")
    l3 = L3Tier(dsn=PG_DSN, table="test_hier_l3")

    await l2.connect()
    await l3.connect()
    await l2.clear()
    await l3.clear()

    h = MemoryHierarchy(tiers=[l1, l2, l3])
    yield h, l1, l2, l3

    await l2.clear()
    await l3.clear()
    await l2.disconnect()
    await l3.disconnect()


async def test_read_from_l3_promotes_to_l1_and_l2(hierarchy):
    h, l1, l2, l3 = hierarchy

    # Write directly to L3, bypassing L1 and L2
    await l3.set("fact:capital", "Paris")

    # First read: should fall through L1 (miss) → L2 (miss) → L3 (hit)
    result = await h.get("fact:capital")
    assert result == "Paris"

    # Value should now be promoted to L1 and L2
    assert await l1.get("fact:capital") == "Paris"
    assert await l2.get("fact:capital") == "Paris"


async def test_second_read_hits_l1(hierarchy):
    h, l1, l2, l3 = hierarchy

    await l3.set("fact:river", "Nile")
    await h.get("fact:river")  # first read — promotes to L1/L2

    # Remove from L3 and L2 to prove L1 is serving the second read
    await l3.delete("fact:river")
    await l2.delete("fact:river")

    result = await h.get("fact:river")
    assert result == "Nile"


async def test_hierarchy_set_and_get_roundtrip(hierarchy):
    h, l1, l2, l3 = hierarchy

    await h.set("key:x", {"value": 42}, tier_index=2)  # write to L3
    result = await h.get("key:x")
    assert result == {"value": 42}


async def test_delete_clears_all_tiers(hierarchy):
    h, l1, l2, l3 = hierarchy

    await h.set("key:del", "to-be-deleted", tier_index=2)
    await h.get("key:del")  # promote to L1/L2
    await h.delete("key:del")

    assert await h.get("key:del") is None

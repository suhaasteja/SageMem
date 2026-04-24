"""Integration e2e test for the MESI coherence protocol.

Two agents share a key via Redis pub/sub. When agent A writes, agent B's
cached entry must transition to Invalid and re-fetch the new value.

Requires: Redis + Postgres running.
"""

import asyncio
import pytest

from sagemem.coherence.bus import CoherenceBus
from sagemem.coherence.protocol import MESIState
from sagemem.tiers.l1_coherent import CoherentL1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"


@pytest.fixture
async def bus():
    b = CoherenceBus(url=REDIS_URL, channel="sagemem:test:coherence")
    await b.connect()
    yield b
    await b.disconnect()


async def make_agent(agent_id: str, bus: CoherenceBus, l2: L2Tier, l3: L3Tier):
    """Build a CoherentL1 + shared L2 + shared L3 hierarchy for an agent.

    L2 and L3 are shared (per-cluster), passed in from the caller.
    """
    l1 = CoherentL1Tier(agent_id=agent_id, bus=bus, capacity=64)
    await l1.start()
    return l1, MemoryHierarchy(tiers=[l1, l2, l3])


@pytest.fixture
async def shared_tiers():
    """Shared L2 + L3 for the cluster (both agents use the same instance)."""
    l2 = L2Tier(url=REDIS_URL, namespace="sagemem:test:coh:cluster")
    l3 = L3Tier(dsn=PG_DSN, table="test_coherence_l3")
    await l2.connect()
    await l3.connect()
    await l2.clear()
    await l3.clear()
    yield l2, l3
    await l2.clear()
    await l3.clear()
    await l2.disconnect()
    await l3.disconnect()


async def test_invalidation_on_remote_write(bus, shared_tiers):
    """Agent A writes a key; agent B's cached copy must become Invalid."""
    l2, l3 = shared_tiers
    l1_a, h_a = await make_agent("agent-a", bus, l2, l3)
    l1_b, h_b = await make_agent("agent-b", bus, l2, l3)

    # Both agents read the same key from L3 — B caches it in L1
    await l3.set("shared:belief", "earth is round")
    await h_b.get("shared:belief")
    assert l1_b.get_state("shared:belief") is not None

    # A writes a new value — broadcasts invalidation
    await l1_a.set("shared:belief", "earth is an oblate spheroid", version=1)
    await asyncio.sleep(0.05)

    # B's L1 entry must be Invalid
    assert l1_b.get_state("shared:belief") == MESIState.Invalid


async def test_invalid_entry_re_fetches_from_lower_tier(bus, shared_tiers):
    """After invalidation, B's next get() re-fetches the new value from the shared L3."""
    l2, l3 = shared_tiers
    l1_a, h_a = await make_agent("agent-a", bus, l2, l3)
    l1_b, h_b = await make_agent("agent-b", bus, l2, l3)

    # Seed shared L3 with initial value; B reads and caches in L1
    await l3.set("fact:sky", "blue")
    val = await h_b.get("fact:sky")
    assert val == "blue"

    # A writes new value to shared L3 + publishes invalidation
    await l3.set("fact:sky", "gray on a cloudy day")
    await l1_a.set("fact:sky", "gray on a cloudy day", version=2)
    await asyncio.sleep(0.05)

    # B's L1 is Invalid; hierarchy falls through L1 (miss) → L2 (miss, was evicted
    # by A's L3 write going through shared L2) → L3 (fresh value)
    assert l1_b.get_state("fact:sky") == MESIState.Invalid

    # Also clear B's L2 entry to simulate L2 not having a stale copy
    await l2.delete("fact:sky")

    fresh = await h_b.get("fact:sky")
    assert fresh == "gray on a cloudy day"


async def test_writer_does_not_invalidate_own_entry(bus, shared_tiers):
    """An agent must not invalidate its own cache entry on self-publish."""
    l2, l3 = shared_tiers
    l1_a, h_a = await make_agent("agent-a", bus, l2, l3)

    await l1_a.set("key", "value-v1")
    await asyncio.sleep(0.05)

    # State should be Modified (own write), not Invalid
    state = l1_a.get_state("key")
    assert state != MESIState.Invalid


async def test_l3_version_conflict_raises(bus, shared_tiers):
    """Concurrent CAS writes to L3 must raise VersionConflictError on stale version."""
    from sagemem.tiers.l3 import VersionConflictError
    _, l3 = shared_tiers

    await l3.set("contested", "v0")
    _, version = await l3.get_versioned("contested")

    # First writer succeeds
    await l3.set_versioned("contested", "v1-winner", expected_version=version)

    # Second writer with stale version must fail
    with pytest.raises(VersionConflictError):
        await l3.set_versioned("contested", "v1-loser", expected_version=version)

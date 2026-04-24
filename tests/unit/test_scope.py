"""Unit tests for AgentCapability and ScopedHierarchy."""

import pytest

from sagemem.tiers.l1 import L1Tier
from sagemem.hierarchy import MemoryHierarchy
from sagemem.scope import AgentCapability, ScopedHierarchy, ScopeViolationError


def make_hierarchy(n: int = 4) -> tuple[MemoryHierarchy, list[L1Tier]]:
    """Build an n-tier in-memory hierarchy for testing."""
    tiers = [L1Tier(capacity=32) for _ in range(n)]
    return MemoryHierarchy(tiers=tiers), tiers


# ---------------------------------------------------------------------------
# AgentCapability — permission checks
# ---------------------------------------------------------------------------

def test_full_capability_can_read_all_tiers():
    cap = AgentCapability(tiers={0, 1, 2, 3})
    for i in range(4):
        assert cap.can_read(i)


def test_full_capability_can_write_all_tiers():
    cap = AgentCapability(tiers={0, 1, 2, 3})
    for i in range(4):
        assert cap.can_write(i)


def test_restricted_to_l1_l2_cannot_access_l3_dram():
    cap = AgentCapability(tiers={0, 1}, l3_read=False, l3_write=False,
                          dram_read=False, dram_write=False)
    assert cap.can_read(0)
    assert cap.can_read(1)
    assert not cap.can_read(2)
    assert not cap.can_read(3)
    assert not cap.can_write(2)
    assert not cap.can_write(3)


def test_read_only_l2():
    cap = AgentCapability(tiers={0, 1, 2, 3}, l2_read=True, l2_write=False)
    assert cap.can_read(1)
    assert not cap.can_write(1)


def test_tier_not_in_set_blocks_access():
    cap = AgentCapability(tiers={0})
    assert not cap.can_read(1)
    assert not cap.can_write(1)


# ---------------------------------------------------------------------------
# ScopedHierarchy — enforcement
# ---------------------------------------------------------------------------

async def test_scoped_write_to_permitted_tier_succeeds():
    h, tiers = make_hierarchy()
    cap = AgentCapability(tiers={0, 1, 2, 3})
    scoped = ScopedHierarchy(h, cap, agent_id="agent-full")
    await scoped.set("k", "v", tier_index=0)
    assert await tiers[0].get("k") == "v"


async def test_scoped_write_to_forbidden_tier_raises():
    h, _ = make_hierarchy()
    cap = AgentCapability(tiers={0, 1}, l3_read=False, l3_write=False,
                          dram_read=False, dram_write=False)
    scoped = ScopedHierarchy(h, cap, agent_id="agent-restricted")
    with pytest.raises(ScopeViolationError, match="agent-restricted"):
        await scoped.set("k", "v", tier_index=2)


async def test_scoped_write_to_tier_not_in_set_raises():
    h, _ = make_hierarchy()
    cap = AgentCapability(tiers={0})
    scoped = ScopedHierarchy(h, cap, agent_id="agent-l1-only")
    with pytest.raises(ScopeViolationError):
        await scoped.set("k", "v", tier_index=1)


async def test_scoped_get_only_reads_permitted_tiers():
    """Data in a forbidden tier must be invisible to a scoped agent."""
    h, tiers = make_hierarchy()
    # Write data directly into tier 2 (L3), bypassing scoped access
    await tiers[2].set("secret", "classified")

    cap = AgentCapability(tiers={0, 1}, l3_read=False, l3_write=False,
                          dram_read=False, dram_write=False)
    scoped = ScopedHierarchy(h, cap, agent_id="agent-restricted")

    # The restricted agent must not see the value
    result = await scoped.get("secret")
    assert result is None


async def test_scoped_get_finds_value_in_permitted_tier():
    h, tiers = make_hierarchy()
    await tiers[1].set("fact", "visible")

    cap = AgentCapability(tiers={0, 1, 2, 3})
    scoped = ScopedHierarchy(h, cap, agent_id="agent-full")

    assert await scoped.get("fact") == "visible"


async def test_scoped_get_promotes_within_permitted_tiers():
    """Promotion must not write to tiers outside the agent's capability."""
    h, tiers = make_hierarchy(4)
    await tiers[2].set("promoted_key", "value")

    # Agent can read tier 2 but can only write to tiers 0 and 1
    cap = AgentCapability(tiers={0, 1, 2, 3}, l3_read=True, l3_write=False,
                          dram_read=True, dram_write=False)
    scoped = ScopedHierarchy(h, cap, agent_id="agent-read-l3")

    result = await scoped.get("promoted_key")
    assert result == "value"
    # Promoted to L1 (writable) and L2 (writable)
    assert await tiers[0].get("promoted_key") == "value"
    assert await tiers[1].get("promoted_key") == "value"


async def test_scoped_exists_only_checks_permitted_tiers():
    h, tiers = make_hierarchy()
    await tiers[3].set("dram_key", "deep")

    cap = AgentCapability(tiers={0, 1}, l3_read=False, l3_write=False,
                          dram_read=False, dram_write=False)
    scoped = ScopedHierarchy(h, cap, agent_id="agent-restricted")

    assert not await scoped.exists("dram_key")


async def test_scoped_delete_only_removes_from_writable_tiers():
    h, tiers = make_hierarchy()
    for t in tiers:
        await t.set("key", "val")

    cap = AgentCapability(tiers={0, 1}, l3_read=False, l3_write=False,
                          dram_read=False, dram_write=False)
    scoped = ScopedHierarchy(h, cap, agent_id="agent-restricted")
    await scoped.delete("key")

    # L1 and L2 deleted
    assert await tiers[0].get("key") is None
    assert await tiers[1].get("key") is None
    # L3 and DRAM untouched (agent can't write there)
    assert await tiers[2].get("key") == "val"
    assert await tiers[3].get("key") == "val"

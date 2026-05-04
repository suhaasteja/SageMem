"""Unit tests for MemoryHierarchy — fall-through read and promotion logic.

Uses in-process L1Tier instances as fakes. No external services needed.
"""

import pytest

from sagemem.tiers.l1 import L1Tier
from sagemem.hierarchy import MemoryHierarchy


def make_hierarchy(*capacities: int) -> tuple[MemoryHierarchy, list[L1Tier]]:
    """Build a hierarchy of N in-memory tiers for testing."""
    tiers = [L1Tier(capacity=c) for c in capacities]
    return MemoryHierarchy(tiers=tiers), tiers


async def test_get_returns_none_when_all_tiers_empty():
    h, _ = make_hierarchy(8, 8, 8)
    assert await h.get("missing") is None


async def test_get_hits_first_tier():
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[0].set("k", "from-l1")
    assert await h.get("k") == "from-l1"


async def test_get_falls_through_to_second_tier():
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[1].set("k", "from-l2")
    assert await h.get("k") == "from-l2"


async def test_get_falls_through_to_third_tier():
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[2].set("k", "from-l3")
    assert await h.get("k") == "from-l3"


async def test_promotion_on_miss():
    """A miss at L1/L2 that hits L3 should promote the value to L1 and L2."""
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[2].set("k", "deep-value")

    result = await h.get("k")
    assert result == "deep-value"

    # Value should now be promoted to both L1 and L2
    assert await tiers[0].get("k") == "deep-value"
    assert await tiers[1].get("k") == "deep-value"


async def test_no_promotion_when_found_in_first_tier():
    """A hit at L1 should not touch L2 or L3."""
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[0].set("k", "l1-value")
    await h.get("k")
    # L2 and L3 remain empty
    assert await tiers[1].get("k") is None
    assert await tiers[2].get("k") is None


async def test_promotion_only_to_faster_tiers():
    """A hit at L2 should promote to L1 only, not write back to L3."""
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[1].set("k", "l2-value")
    await h.get("k")
    assert await tiers[0].get("k") == "l2-value"   # promoted
    assert await tiers[2].get("k") is None           # untouched


async def test_set_writes_to_correct_tier():
    h, tiers = make_hierarchy(8, 8, 8)
    await h.set("k", "v", tier_index=1)
    assert await tiers[1].get("k") == "v"
    assert await tiers[0].get("k") is None
    assert await tiers[2].get("k") is None


async def test_set_defaults_to_tier_zero():
    h, tiers = make_hierarchy(8, 8, 8)
    await h.set("k", "v")
    assert await tiers[0].get("k") == "v"


async def test_set_invalid_tier_raises():
    h, _ = make_hierarchy(8)
    with pytest.raises(IndexError):
        await h.set("k", "v", tier_index=5)


async def test_delete_removes_from_all_tiers():
    h, tiers = make_hierarchy(8, 8, 8)
    for t in tiers:
        await t.set("k", "v")
    await h.delete("k")
    for t in tiers:
        assert await t.get("k") is None


async def test_second_get_hits_l1_after_promotion():
    """After promotion, a second read should hit L1 directly."""
    h, tiers = make_hierarchy(8, 8, 8)
    await tiers[2].set("k", "v")

    await h.get("k")           # miss L1, miss L2, hit L3 → promote
    await tiers[2].delete("k") # remove from L3 to prove L1 is serving it

    assert await h.get("k") == "v"  # served from L1


async def test_clear_removes_from_all_tiers():
    """clear() should empty every tier in the hierarchy."""
    h, tiers = make_hierarchy(8, 8, 8)
    for i, t in enumerate(tiers):
        await t.set(f"k{i}", f"v{i}")

    await h.clear()

    for i, t in enumerate(tiers):
        assert await t.get(f"k{i}") is None


async def test_semantic_search_delegates_to_tier_with_embedder():
    """semantic_search() should use the last tier that has search + embedder."""
    import asyncio

    recorded_embeddings: list = []

    async def fake_embedder(text: str) -> list[float]:
        recorded_embeddings.append(text)
        return [0.1, 0.2, 0.3]

    class FakeDRAM(L1Tier):
        def __init__(self):
            super().__init__(capacity=8)
            self.embedder = fake_embedder

        async def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
            return [{"value": {"speaker": "Alice", "text": "hello"}, "distance": 0.1}]

    dram = FakeDRAM()
    l1 = L1Tier(capacity=8)
    h = MemoryHierarchy(tiers=[l1, dram])

    results = await h.semantic_search("find something", top_k=3)

    assert len(results) == 1
    assert results[0]["speaker"] == "Alice"
    assert recorded_embeddings == ["find something"]


async def test_semantic_search_returns_empty_when_no_capable_tier():
    """semantic_search() returns [] when no tier has search + embedder."""
    h, _ = make_hierarchy(8, 8)
    results = await h.semantic_search("anything")
    assert results == []

"""MemoryHierarchy — composes tiers into a unified read/write interface.

Read path: L1 → L2 → L3 → DRAM, promoting on miss to the fastest tier that saw the miss.
Write path: writes to a specified tier index only (callers decide where data lives).
"""

import time
from typing import Any

from sagemem.tiers.base import Tier
from sagemem.metrics import (
    tier_hits,
    tier_misses,
    tier_promotions,
    tier_read_latency,
    tier_write_latency,
)

TIER_NAMES = ["l1", "l2", "l3", "dram"]


def _tier_name(index: int) -> str:
    """Return a label for tier at index, falling back to 'tier_N'."""
    if index < len(TIER_NAMES):
        return TIER_NAMES[index]
    return f"tier_{index}"


class MemoryHierarchy:
    """Ordered stack of tiers. Reads fall through L1 → L2 → L3 → DRAM.

    On a miss at tier N, the value is fetched from the next tier and
    promoted back to all tiers faster than where it was found.

    Args:
        tiers: Ordered list of Tier instances, fastest (L1) first.
    """

    def __init__(self, tiers: list[Tier]) -> None:
        """Initialize hierarchy with an ordered list of tiers."""
        self.tiers = tiers

    async def get(self, key: str) -> Any | None:
        """Read key, falling through tiers on miss and promoting on find.

        Returns the value if found in any tier, None otherwise.
        Emits hit/miss/promotion metrics.
        """
        for i, tier in enumerate(self.tiers):
            name = _tier_name(i)
            t0 = time.perf_counter()
            value = await tier.get(key)
            tier_read_latency.labels(tier=name).observe(time.perf_counter() - t0)

            if value is not None:
                tier_hits.labels(tier=name).inc()
                # Promote to all faster tiers (those before index i)
                for j in range(i):
                    faster_name = _tier_name(j)
                    t1 = time.perf_counter()
                    await self.tiers[j].set(key, value)
                    tier_write_latency.labels(tier=faster_name).observe(time.perf_counter() - t1)
                    tier_promotions.labels(from_tier=name, to_tier=faster_name).inc()
                return value

            tier_misses.labels(tier=name).inc()

        return None

    async def set(self, key: str, value: Any, tier_index: int = 0) -> None:
        """Write value to the tier at tier_index.

        Args:
            key: Cache key.
            value: Value to store.
            tier_index: Which tier to write to (0 = L1, 1 = L2, etc.).
        """
        if not self.tiers:
            return
        if tier_index >= len(self.tiers):
            raise IndexError(
                f"tier_index {tier_index} out of range — hierarchy has {len(self.tiers)} tiers."
            )
        name = _tier_name(tier_index)
        t0 = time.perf_counter()
        await self.tiers[tier_index].set(key, value)
        tier_write_latency.labels(tier=name).observe(time.perf_counter() - t0)

    async def delete(self, key: str) -> None:
        """Delete key from all tiers."""
        for tier in self.tiers:
            await tier.delete(key)

    async def exists_in(self, key: str, tier_index: int) -> bool:
        """Return True if key exists in the tier at tier_index."""
        if tier_index >= len(self.tiers):
            raise IndexError(f"tier_index {tier_index} out of range.")
        return await self.tiers[tier_index].exists(key)

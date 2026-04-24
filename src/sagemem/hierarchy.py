"""MemoryHierarchy — composes tiers into a unified read/write interface.

Stage 0: stubbed. Read path (fall-through + promotion) implemented in Stage 2.
"""

from typing import Any

from sagemem.tiers.base import Tier


class MemoryHierarchy:
    """Ordered stack of tiers. Reads fall through L1 → L2 → L3 → DRAM.

    Args:
        tiers: Ordered list of tiers, fastest first.
    """

    def __init__(self, tiers: list[Tier]) -> None:
        """Initialize hierarchy with an ordered list of tiers."""
        self.tiers = tiers

    async def get(self, key: str) -> Any | None:
        """Read key from the fastest available tier, promoting on miss."""
        raise NotImplementedError("Hierarchical read path implemented in Stage 2.")

    async def set(self, key: str, value: Any, tier_index: int = 0) -> None:
        """Write value to the specified tier."""
        raise NotImplementedError("Write path implemented in Stage 2.")

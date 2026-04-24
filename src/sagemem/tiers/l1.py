"""L1 tier — per-agent in-memory LRU cache."""

from collections import OrderedDict
from typing import Any

from sagemem.tiers.base import Tier


class L1Tier(Tier):
    """In-memory LRU cache. Fastest tier, per-agent, bounded by capacity.

    Args:
        capacity: Maximum number of entries before LRU eviction occurs.
    """

    def __init__(self, capacity: int = 128) -> None:
        """Initialize with a fixed capacity."""
        self.capacity = capacity
        self._store: OrderedDict[str, Any] = OrderedDict()

    async def get(self, key: str) -> Any | None:
        """Return value for key and mark it as recently used, or None."""
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    async def set(self, key: str, value: Any) -> None:
        """Store value under key, evicting the LRU entry if at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict least recently used

    async def delete(self, key: str) -> None:
        """Remove key from the cache. No-op if not present."""
        self._store.pop(key, None)

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    async def exists(self, key: str) -> bool:
        """Return True if key is present."""
        return key in self._store

    def __len__(self) -> int:
        """Return number of entries currently in cache."""
        return len(self._store)

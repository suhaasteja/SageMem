"""CoherentL1Tier — MESI-aware in-memory cache wired to the CoherenceBus.

Extends L1Tier with:
- CacheEntry tracking (state + version) per key
- Publishes invalidations on write to Shared/Exclusive entries
- Receives invalidations from the bus and marks entries Invalid
- On get of an Invalid entry, forces a miss so the hierarchy re-fetches
"""

import logging
from collections import OrderedDict
from typing import Any

from sagemem.coherence.bus import CoherenceBus, InvalidateMessage
from sagemem.coherence.protocol import CacheEntry, MESIState, apply_event
from sagemem.tiers.base import Tier

logger = logging.getLogger(__name__)


class CoherentL1Tier(Tier):
    """MESI-aware in-memory LRU cache.

    Args:
        agent_id: Unique identifier for this agent (used in invalidation messages).
        bus: CoherenceBus instance shared across agents.
        capacity: Maximum number of entries before LRU eviction.
    """

    def __init__(self, agent_id: str, bus: CoherenceBus, capacity: int = 128) -> None:
        """Initialize with agent identity, shared bus, and capacity."""
        self.agent_id = agent_id
        self.bus = bus
        self.capacity = capacity
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    async def start(self) -> None:
        """Subscribe to the coherence bus. Must be called after bus.connect()."""
        await self.bus.subscribe(self.agent_id, self._on_invalidate)

    async def _on_invalidate(self, msg: InvalidateMessage) -> None:
        """Handle an incoming invalidation message from another agent."""
        if msg.writer_id == self.agent_id:
            return  # ignore our own publishes
        if msg.key in self._store:
            entry = self._store[msg.key]
            apply_event(entry, "remote_write")
            logger.debug("[%s] invalidated key=%r new_state=%s",
                         self.agent_id, msg.key, entry.state.name)

    async def get(self, key: str) -> Any | None:
        """Return value if present and not Invalid; None otherwise (forces re-fetch)."""
        if key not in self._store:
            return None
        entry = self._store[key]
        if entry.state == MESIState.Invalid:
            # Stale — treat as miss so the hierarchy re-fetches from a lower tier
            logger.debug("[%s] get key=%r state=Invalid → miss", self.agent_id, key)
            return None
        self._store.move_to_end(key)
        apply_event(entry, "local_read")
        return entry.value

    async def set(self, key: str, value: Any, version: int = 0) -> None:
        """Store value and broadcast invalidation to all other agents.

        Always publishes — the writer doesn't need to know who else has the
        key cached. That's the point of broadcast invalidation.
        """
        existing = self._store.get(key)
        if existing is not None:
            apply_event(existing, "local_write")
            existing.value = value
            existing.version = version
            self._store.move_to_end(key)
        else:
            new_entry = CacheEntry(key=key, value=value, state=MESIState.Modified, version=version)
            self._store[key] = new_entry
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)

        # Always broadcast — other agents may have a stale cached copy
        msg = InvalidateMessage(key=key, writer_id=self.agent_id, new_version=version)
        await self.bus.publish_invalidate(msg)
        logger.debug("[%s] published invalidate key=%r", self.agent_id, key)

    async def set_shared(self, key: str, value: Any, version: int = 0) -> None:
        """Store value promoted from a shared tier — marks entry as Shared."""
        entry = CacheEntry(key=key, value=value, state=MESIState.Shared, version=version)
        self._store[key] = entry
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        """Remove key from the cache."""
        self._store.pop(key, None)

    async def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    async def exists(self, key: str) -> bool:
        """Return True if key is present and not Invalid."""
        entry = self._store.get(key)
        return entry is not None and entry.state != MESIState.Invalid

    def get_state(self, key: str) -> MESIState | None:
        """Return the MESI state of a key, or None if not present."""
        entry = self._store.get(key)
        return entry.state if entry else None

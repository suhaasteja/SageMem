"""MESI coherence protocol state definitions."""

from enum import Enum, auto


class MESIState(Enum):
    """Cache line states for the MESI coherence protocol.

    Modified  — local has a write not yet propagated to shared tiers.
    Exclusive — local matches shared tier; no other agent has this key cached.
    Shared    — local matches shared tier; other agents may also have it cached.
    Invalid   — entry is stale; must re-fetch from the next tier down.
    """

    Modified = auto()
    Exclusive = auto()
    Shared = auto()
    Invalid = auto()


class CacheEntry:
    """A single cached value with its MESI state and version."""

    def __init__(self, key: str, value: object, state: MESIState, version: int = 0) -> None:
        """Initialize a cache entry."""
        self.key = key
        self.value = value
        self.state = state
        self.version = version

    def __repr__(self) -> str:
        return f"CacheEntry(key={self.key!r}, state={self.state.name}, version={self.version})"

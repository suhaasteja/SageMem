"""MESI coherence protocol — state machine and cache entry management."""

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


# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------
# Each entry is (from_state, event) -> to_state
# Events:
#   local_read   — this agent reads the key
#   local_write  — this agent writes the key
#   remote_write — another agent wrote this key (invalidation received)
#   evict        — entry is evicted from local cache
#   fetch        — value re-fetched from shared tier after Invalid

_TRANSITIONS: dict[tuple[MESIState, str], MESIState] = {
    # Modified: we own the value exclusively with a pending write
    (MESIState.Modified,   "local_read"):   MESIState.Modified,
    (MESIState.Modified,   "local_write"):  MESIState.Modified,
    (MESIState.Modified,   "remote_write"): MESIState.Invalid,
    (MESIState.Modified,   "evict"):        MESIState.Invalid,

    # Exclusive: we have a clean copy, no other cacher
    (MESIState.Exclusive,  "local_read"):   MESIState.Exclusive,
    (MESIState.Exclusive,  "local_write"):  MESIState.Modified,
    (MESIState.Exclusive,  "remote_write"): MESIState.Invalid,
    (MESIState.Exclusive,  "evict"):        MESIState.Invalid,

    # Shared: clean copy, other agents may also hold it
    (MESIState.Shared,     "local_read"):   MESIState.Shared,
    (MESIState.Shared,     "local_write"):  MESIState.Modified,
    (MESIState.Shared,     "remote_write"): MESIState.Invalid,
    (MESIState.Shared,     "evict"):        MESIState.Invalid,

    # Invalid: stale; must re-fetch before use
    (MESIState.Invalid,    "fetch"):        MESIState.Shared,
    (MESIState.Invalid,    "local_write"):  MESIState.Modified,
}


def transition(state: MESIState, event: str) -> MESIState:
    """Return the next MESI state given the current state and an event.

    Raises ValueError for undefined transitions.
    """
    key = (state, event)
    if key not in _TRANSITIONS:
        raise ValueError(f"No transition defined for state={state.name!r} event={event!r}")
    return _TRANSITIONS[key]


def apply_event(entry: CacheEntry, event: str) -> CacheEntry:
    """Apply an event to a CacheEntry in place, returning the entry.

    Updates entry.state according to the MESI state machine.
    """
    entry.state = transition(entry.state, event)
    return entry

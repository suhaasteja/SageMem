"""Unit tests for the MESI state machine — all valid transitions."""

import pytest

from sagemem.coherence.protocol import MESIState, CacheEntry, transition, apply_event


# ---------------------------------------------------------------------------
# transition() — table of valid transitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,event,expected", [
    # Modified
    (MESIState.Modified,  "local_read",   MESIState.Modified),
    (MESIState.Modified,  "local_write",  MESIState.Modified),
    (MESIState.Modified,  "remote_write", MESIState.Invalid),
    (MESIState.Modified,  "evict",        MESIState.Invalid),
    # Exclusive
    (MESIState.Exclusive, "local_read",   MESIState.Exclusive),
    (MESIState.Exclusive, "local_write",  MESIState.Modified),
    (MESIState.Exclusive, "remote_write", MESIState.Invalid),
    (MESIState.Exclusive, "evict",        MESIState.Invalid),
    # Shared
    (MESIState.Shared,    "local_read",   MESIState.Shared),
    (MESIState.Shared,    "local_write",  MESIState.Modified),
    (MESIState.Shared,    "remote_write", MESIState.Invalid),
    (MESIState.Shared,    "evict",        MESIState.Invalid),
    # Invalid
    (MESIState.Invalid,   "fetch",        MESIState.Shared),
    (MESIState.Invalid,   "local_write",  MESIState.Modified),
])
def test_valid_transition(state, event, expected):
    assert transition(state, event) == expected


def test_undefined_transition_raises():
    with pytest.raises(ValueError, match="No transition"):
        transition(MESIState.Modified, "fetch")


def test_undefined_event_raises():
    with pytest.raises(ValueError):
        transition(MESIState.Shared, "bogus_event")


# ---------------------------------------------------------------------------
# apply_event() — mutates CacheEntry in place
# ---------------------------------------------------------------------------

def test_apply_event_mutates_state():
    entry = CacheEntry(key="k", value="v", state=MESIState.Exclusive)
    apply_event(entry, "local_write")
    assert entry.state == MESIState.Modified


def test_apply_event_returns_entry():
    entry = CacheEntry(key="k", value="v", state=MESIState.Shared)
    result = apply_event(entry, "local_read")
    assert result is entry


# ---------------------------------------------------------------------------
# Multi-step transition sequences
# ---------------------------------------------------------------------------

def test_write_to_shared_becomes_modified_then_invalid():
    """Shared → local_write → Modified → remote_write → Invalid."""
    entry = CacheEntry(key="k", value="v", state=MESIState.Shared)
    apply_event(entry, "local_write")
    assert entry.state == MESIState.Modified
    apply_event(entry, "remote_write")
    assert entry.state == MESIState.Invalid


def test_invalid_fetch_then_write():
    """Invalid → fetch → Shared → local_write → Modified."""
    entry = CacheEntry(key="k", value="v", state=MESIState.Invalid)
    apply_event(entry, "fetch")
    assert entry.state == MESIState.Shared
    apply_event(entry, "local_write")
    assert entry.state == MESIState.Modified


def test_exclusive_read_stays_exclusive():
    entry = CacheEntry(key="k", value="v", state=MESIState.Exclusive)
    for _ in range(5):
        apply_event(entry, "local_read")
    assert entry.state == MESIState.Exclusive


def test_evict_from_any_state_goes_invalid():
    for state in (MESIState.Modified, MESIState.Exclusive, MESIState.Shared):
        entry = CacheEntry(key="k", value="v", state=state)
        apply_event(entry, "evict")
        assert entry.state == MESIState.Invalid

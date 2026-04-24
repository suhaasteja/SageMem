"""Hypothesis property tests for the MESI coherence protocol.

Tests the invariants that must hold regardless of event sequence:
- No undefined state ever appears
- After remote_write, state is always Invalid
- After local_write, state is always Modified
- Convergence: two agents writing concurrently always end in a consistent state
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sagemem.coherence.protocol import MESIState, CacheEntry, transition, apply_event

# All valid events per state
_VALID_EVENTS: dict[MESIState, list[str]] = {
    MESIState.Modified:  ["local_read", "local_write", "remote_write", "evict"],
    MESIState.Exclusive: ["local_read", "local_write", "remote_write", "evict"],
    MESIState.Shared:    ["local_read", "local_write", "remote_write", "evict"],
    MESIState.Invalid:   ["fetch", "local_write"],
}

ALL_STATES = list(MESIState)


def valid_event_for(state: MESIState) -> st.SearchStrategy:
    """Strategy that yields only events valid for the given state."""
    return st.sampled_from(_VALID_EVENTS[state])


@given(
    initial=st.sampled_from(ALL_STATES),
    events=st.lists(st.text(min_size=1), min_size=0, max_size=10),
)
def test_undefined_events_always_raise(initial, events):
    """Any event not in _VALID_EVENTS for the current state must raise ValueError."""
    state = initial
    for event in events:
        valid = _VALID_EVENTS[state]
        if event not in valid:
            try:
                transition(state, event)
                assert False, f"Expected ValueError for state={state.name} event={event!r}"
            except ValueError:
                pass  # correct
            return  # stop after first invalid event


@given(
    initial=st.sampled_from(ALL_STATES),
    n=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=200)
def test_state_always_remains_valid_mesi_state(initial, n):
    """Applying any sequence of valid events always yields a valid MESIState."""
    entry = CacheEntry(key="k", value="v", state=initial)
    for _ in range(n):
        event = _VALID_EVENTS[entry.state][0]  # always pick the first valid event
        apply_event(entry, event)
        assert entry.state in ALL_STATES


@given(st.sampled_from([MESIState.Modified, MESIState.Exclusive, MESIState.Shared]))
def test_remote_write_always_invalidates(state):
    """remote_write from any non-Invalid state must yield Invalid."""
    entry = CacheEntry(key="k", value="v", state=state)
    apply_event(entry, "remote_write")
    assert entry.state == MESIState.Invalid


@given(st.sampled_from(ALL_STATES))
def test_local_write_always_yields_modified(state):
    """local_write from any state that supports it must yield Modified."""
    assume("local_write" in _VALID_EVENTS[state])
    entry = CacheEntry(key="k", value="v", state=state)
    apply_event(entry, "local_write")
    assert entry.state == MESIState.Modified


@given(
    initial=st.sampled_from(ALL_STATES),
    event_indices=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=30),
)
@settings(max_examples=500)
def test_no_transition_produces_unknown_state(initial, event_indices):
    """Exhaustive: no sequence of valid events ever produces an unknown state."""
    entry = CacheEntry(key="k", value="v", state=initial)
    for idx in event_indices:
        valid = _VALID_EVENTS[entry.state]
        event = valid[idx % len(valid)]
        apply_event(entry, event)
        assert entry.state in set(MESIState), f"Unknown state: {entry.state}"


@given(
    state_a=st.sampled_from(ALL_STATES),
    state_b=st.sampled_from(ALL_STATES),
)
def test_concurrent_write_both_converge_to_modified_or_invalid(state_a, state_b):
    """Two agents writing the same key: each ends in Modified or Invalid.

    After agent A writes (local_write), agent B receives remote_write.
    Both must be in a deterministic state — no stuck or undefined state.
    """
    assume("local_write" in _VALID_EVENTS[state_a])

    entry_a = CacheEntry(key="k", value="v1", state=state_a)
    entry_b = CacheEntry(key="k", value="v2", state=state_b)

    # A writes
    apply_event(entry_a, "local_write")
    assert entry_a.state == MESIState.Modified

    # B receives invalidation (if it had the entry)
    if "remote_write" in _VALID_EVENTS[entry_b.state]:
        apply_event(entry_b, "remote_write")
        assert entry_b.state == MESIState.Invalid

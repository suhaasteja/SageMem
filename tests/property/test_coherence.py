"""Hypothesis property tests for the MESI coherence protocol.

Tests the invariants that must hold regardless of event sequence:
- No undefined state ever appears
- After remote_write, state is always Invalid
- After local_write, state is always Modified
- Convergence: two agents writing concurrently always end in a consistent state

Global MESI invariants (multi-agent):
- If any agent holds Modified, all others must be Invalid for that key
- No agent can hold Shared if another holds Modified for the same key
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


# ---------------------------------------------------------------------------
# Global MESI invariants — multi-agent simulations
# ---------------------------------------------------------------------------
# These test the system-level properties that local state machine tests miss:
# "if any cache holds Modified, no other cache may hold anything but Invalid."


def _simulate_agent_sequence(n_agents: int, event_indices: list[int]) -> list[CacheEntry]:
    """Simulate N agents each processing one event from a shared sequence.

    Each agent starts Exclusive (clean copy, sole owner).
    Events are chosen round-robin across agents from event_indices.

    Two protocol rules are enforced:
    1. Write (→ Modified): all other agents receive remote_write (→ Invalid).
    2. Fetch (Invalid → Shared): any Modified holder first flushes to Shared
       (write-back), modelling the MESI bus-snoop / write-back-on-read protocol.

    Returns the final list of CacheEntry objects, one per agent.
    """
    entries = [
        CacheEntry(key="shared-key", value=f"agent-{i}", state=MESIState.Exclusive)
        for i in range(n_agents)
    ]

    for step, idx in enumerate(event_indices):
        actor = step % n_agents
        current_state = entries[actor].state
        valid = _VALID_EVENTS[current_state]
        event = valid[idx % len(valid)]

        # Write-back-on-fetch: before a fetch completes, any Modified holder must
        # flush (Modified → Shared) so the shared tier is coherent.
        if event == "fetch":
            for j, other in enumerate(entries):
                if j != actor and other.state == MESIState.Modified:
                    other.state = MESIState.Shared  # writeback to shared tier

        apply_event(entries[actor], event)

        # Write propagation: if actor wrote (→ Modified), invalidate all others.
        if entries[actor].state == MESIState.Modified:
            for j, other in enumerate(entries):
                if j != actor and "remote_write" in _VALID_EVENTS[other.state]:
                    apply_event(other, "remote_write")

    return entries


@given(
    n_agents=st.integers(min_value=2, max_value=6),
    event_indices=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=50),
)
@settings(max_examples=300)
def test_modified_implies_all_others_invalid(n_agents, event_indices):
    """Global invariant: if any agent holds Modified, every other must be Invalid.

    This is the real MESI invariant — not just per-entry state transitions.
    """
    entries = _simulate_agent_sequence(n_agents, event_indices)

    modified = [e for e in entries if e.state == MESIState.Modified]
    if not modified:
        return  # invariant vacuously satisfied

    assert len(modified) == 1, (
        f"More than one agent in Modified state: {[e for e in entries]}"
    )
    others = [e for e in entries if e.state != MESIState.Modified]
    for other in others:
        assert other.state == MESIState.Invalid, (
            f"Expected Invalid but got {other.state.name} while another agent is Modified"
        )


@given(
    n_agents=st.integers(min_value=2, max_value=6),
    event_indices=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=50),
)
@settings(max_examples=300)
def test_shared_never_coexists_with_modified(n_agents, event_indices):
    """Global invariant: no agent can be Shared while another is Modified."""
    entries = _simulate_agent_sequence(n_agents, event_indices)

    has_modified = any(e.state == MESIState.Modified for e in entries)
    has_shared = any(e.state == MESIState.Shared for e in entries)

    assert not (has_modified and has_shared), (
        f"Shared and Modified coexist: {[(e.state.name) for e in entries]}"
    )

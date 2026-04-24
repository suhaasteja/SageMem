"""Demo scenario: two agents write conflicting beliefs, coherence resolves them.

Yields structured events consumed by the SSE stream.
"""

import asyncio
from typing import AsyncIterator


async def run_scenario(emit) -> None:
    """Execute the demo scenario, calling emit(event_type, data) for each step."""

    async def step(event_type: str, data: dict, delay: float = 0.8):
        await emit(event_type, data)
        await asyncio.sleep(delay)

    # --- Setup ---
    await step("status", {"msg": "Initializing agents and memory hierarchy..."})
    await step("agent_state", {"agent": "A", "tier": "L1", "state": "empty", "key": None, "value": None})
    await step("agent_state", {"agent": "B", "tier": "L1", "state": "empty", "key": None, "value": None})
    await step("tier_state",  {"tier": "L2", "key": None, "value": None, "state": "empty"})
    await step("tier_state",  {"tier": "L3", "key": None, "value": None, "state": "empty"})

    # --- Agent A seeds a belief in L3 ---
    await step("status", {"msg": "Agent A stores initial belief in L3..."})
    await step("tier_state", {"tier": "L3", "key": "belief:climate", "value": "CO₂ is the primary driver", "state": "written"})
    await step("agent_state", {"agent": "A", "tier": "L1", "state": "exclusive", "key": "belief:climate", "value": "CO₂ is the primary driver", "mesi": "Exclusive"})

    # --- Agent B reads → cache miss → promotes from L3 ---
    await step("status", {"msg": "Agent B reads 'belief:climate' — L1 miss, L3 hit..."}, delay=1.0)
    await step("event", {"type": "miss", "agent": "B", "tier": "L1", "key": "belief:climate"})
    await step("event", {"type": "miss", "agent": "B", "tier": "L2", "key": "belief:climate"})
    await step("event", {"type": "hit",  "agent": "B", "tier": "L3", "key": "belief:climate"})
    await step("event", {"type": "promote", "agent": "B", "from_tier": "L3", "to_tier": "L1", "key": "belief:climate"})
    await step("agent_state", {"agent": "B", "tier": "L1", "state": "shared", "key": "belief:climate", "value": "CO₂ is the primary driver", "mesi": "Shared"})

    # --- Agent A writes a conflicting value ---
    await step("status", {"msg": "Agent A writes conflicting belief — triggers invalidation..."}, delay=1.2)
    await step("agent_state", {"agent": "A", "tier": "L1", "state": "modified", "key": "belief:climate", "value": "Solar activity is primary driver", "mesi": "Modified"})
    await step("tier_state",  {"tier": "L3", "key": "belief:climate", "value": "Solar activity is primary driver", "state": "written"})

    # --- Invalidation message fires ---
    await step("event", {"type": "invalidate", "from_agent": "A", "to_agent": "B", "key": "belief:climate", "version": 2})
    await step("agent_state", {"agent": "B", "tier": "L1", "state": "invalid", "key": "belief:climate", "value": "CO₂ is the primary driver", "mesi": "Invalid"})

    # --- B re-fetches the fresh value ---
    await step("status", {"msg": "Agent B detects stale entry → re-fetches from L3..."}, delay=1.0)
    await step("event", {"type": "refetch", "agent": "B", "tier": "L3", "key": "belief:climate"})
    await step("agent_state", {"agent": "B", "tier": "L1", "state": "shared", "key": "belief:climate", "value": "Solar activity is primary driver", "mesi": "Shared"})

    # --- Convergence ---
    await step("status", {"msg": "Both agents now hold the same value — coherence achieved ✓"}, delay=1.5)
    await step("converged", {"key": "belief:climate", "value": "Solar activity is primary driver"})

    # --- L3 version conflict demo ---
    await step("status", {"msg": "Simulating concurrent L3 writes (optimistic CAS)..."}, delay=1.0)
    await step("event", {"type": "cas_attempt", "agent": "A", "key": "belief:climate", "expected_version": 1})
    await step("event", {"type": "cas_attempt", "agent": "B", "key": "belief:climate", "expected_version": 1})
    await step("event", {"type": "cas_win",    "agent": "A", "key": "belief:climate"})
    await step("event", {"type": "cas_conflict","agent": "B", "key": "belief:climate", "msg": "VersionConflictError — B retries"})
    await step("status", {"msg": "Demo complete. Refresh to replay."}, delay=0.5)
    await emit("done", {})

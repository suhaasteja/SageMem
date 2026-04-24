"""Parallel search demo — two agents with different tier scopes.

Agent Alpha: full access (L1+L2+L3) — can read and write globally
Agent Beta:  restricted (L1+L2 only) — cannot see L3 data

Demonstrates that ScopedHierarchy is a hard permission boundary, not a hint.

Run with:
    uv run python examples/parallel_search.py

No ANTHROPIC_API_KEY needed — this demo doesn't call the LLM.
"""

import asyncio

from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy
from sagemem.scope import AgentCapability, ScopedHierarchy, ScopeViolationError

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"


def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


async def main() -> None:
    # --- shared infrastructure ---
    l1_alpha = L1Tier(capacity=64)
    l1_beta  = L1Tier(capacity=32)   # smaller budget for beta
    l2 = L2Tier(url=REDIS_URL, namespace="sagemem:demo:parallel")
    l3 = L3Tier(dsn=PG_DSN, table="demo_parallel_l3")

    await l2.connect()
    await l3.connect()
    await l2.clear()
    await l3.clear()

    # Alpha: full access (L1+L2+L3)
    h_alpha = MemoryHierarchy(tiers=[l1_alpha, l2, l3])
    cap_alpha = AgentCapability(
        tiers={0, 1, 2},
        l1_budget_tokens=8000,
        l2_read=True, l2_write=True,
        l3_read=True, l3_write=True,
    )
    alpha = ScopedHierarchy(h_alpha, cap_alpha, agent_id="alpha")

    # Beta: L1+L2 only — cannot access L3
    h_beta = MemoryHierarchy(tiers=[l1_beta, l2, l3])
    cap_beta = AgentCapability(
        tiers={0, 1},
        l1_budget_tokens=4000,
        l2_read=True, l2_write=True,
        l3_read=False, l3_write=False,
    )
    beta = ScopedHierarchy(h_beta, cap_beta, agent_id="beta")

    # --- Demo ---
    separator("Alpha writes a sensitive finding to L3")
    await alpha.set("finding:classified", "dark matter density = 0.3 GeV/cm³", tier_index=2)
    print("  alpha → set 'finding:classified' in L3 ✓")

    separator("Alpha writes a shared result to L2 (visible to both)")
    await alpha.set("finding:shared", "galaxy rotation curves are anomalous", tier_index=1)
    print("  alpha → set 'finding:shared' in L2 ✓")

    separator("Beta reads shared result from L2")
    val = await beta.get("finding:shared")
    print(f"  beta  → get 'finding:shared': {val!r} ✓")

    separator("Beta attempts to read classified L3 data")
    val = await beta.get("finding:classified")
    print(f"  beta  → get 'finding:classified': {val!r}  ← None (scope blocks L3 read)")

    separator("Beta attempts to write directly to L3")
    try:
        await beta.set("finding:beta_attempt", "some data", tier_index=2)
        print("  ERROR: should have raised ScopeViolationError")
    except ScopeViolationError as e:
        print(f"  beta  → ScopeViolationError raised ✓")
        print(f"          {e}")

    separator("Alpha can still read its own L3 data")
    val = await alpha.get("finding:classified")
    print(f"  alpha → get 'finding:classified': {val!r} ✓")

    separator("Summary")
    print("  Alpha (L1+L2+L3): read/write to all three tiers")
    print("  Beta  (L1+L2):    read/write to L1+L2; L3 is physically inaccessible")
    print("  Scope enforcement is hard — not advisory")
    print()

    await l2.clear()
    await l3.clear()
    await l2.disconnect()
    await l3.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

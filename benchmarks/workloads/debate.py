"""Debate workload — two agents writing conflicting beliefs to shared keys.

Agent A and B repeatedly write their own version of shared beliefs.
Each write triggers an invalidation; the other agent must re-fetch on next read.

Measures: coherence invalidation count, convergence behavior.
"""

import asyncio
import time

from sagemem.coherence.bus import CoherenceBus
from sagemem.tiers.l1_coherent import CoherentL1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy
from benchmarks.common import BenchmarkStats

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"

BELIEFS = [
    "belief:origin_of_universe",
    "belief:nature_of_consciousness",
    "belief:optimal_diet",
    "belief:free_will",
    "belief:best_programming_language",
]

ROUNDS = 5


async def run(stats: BenchmarkStats) -> None:
    """Run the debate workload and populate stats."""
    bus = CoherenceBus(url=REDIS_URL, channel="bench:debate:bus")
    shared_l2 = L2Tier(url=REDIS_URL, namespace="bench:debate:l2")
    shared_l3 = L3Tier(dsn=PG_DSN, table="bench_debate_l3")

    await bus.connect()
    await shared_l2.connect()
    await shared_l3.connect()
    await shared_l2.clear()
    await shared_l3.clear()

    l1_a = CoherentL1Tier(agent_id="debater-a", bus=bus, capacity=32)
    l1_b = CoherentL1Tier(agent_id="debater-b", bus=bus, capacity=32)
    await l1_a.start()
    await l1_b.start()

    h_a = MemoryHierarchy(tiers=[l1_a, shared_l2, shared_l3])
    h_b = MemoryHierarchy(tiers=[l1_b, shared_l2, shared_l3])

    invalidation_count = 0
    original_publish = bus.publish_invalidate

    async def counting_publish(msg):
        nonlocal invalidation_count
        invalidation_count += 1
        await original_publish(msg)

    bus.publish_invalidate = counting_publish

    t0 = time.perf_counter()

    for round_num in range(ROUNDS):
        # A writes its position on all beliefs
        for belief in BELIEFS:
            await h_a.set(belief, f"A-position-round-{round_num}", tier_index=0)
            stats.total_writes += 1

        await asyncio.sleep(0.02)  # let invalidations propagate

        # B reads (triggers re-fetch if invalidated) then writes its own position
        for belief in BELIEFS:
            await h_b.get(belief)
            stats.total_reads += 1
            await h_b.set(belief, f"B-position-round-{round_num}", tier_index=0)
            stats.total_writes += 1

        await asyncio.sleep(0.02)

    stats.duration_ms = (time.perf_counter() - t0) * 1000
    stats.coherence_invalidations = invalidation_count

    await shared_l2.clear()
    await shared_l3.clear()
    await shared_l2.disconnect()
    await shared_l3.disconnect()
    await bus.disconnect()

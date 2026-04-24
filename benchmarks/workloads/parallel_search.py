"""Parallel search workload — N agents search and share findings.

Each agent stores findings in L2 (shared cluster tier). Later agents
get L2 hits on keys already discovered by earlier agents, avoiding
redundant L3 reads.

Measures: L2 cross-agent hit rate, total L3 reads avoided.
"""

import asyncio
import time

from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy
from benchmarks.common import BenchmarkStats

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"

N_AGENTS = 4
KEYS_PER_AGENT = 8

# Simulated corpus: each agent "finds" the same set of keys (realistic overlap)
CORPUS = [f"doc:{i:03d}" for i in range(KEYS_PER_AGENT * 2)]


async def run(stats: BenchmarkStats) -> None:
    """Run the parallel search workload and populate stats."""
    shared_l2 = L2Tier(url=REDIS_URL, namespace="bench:parallel:l2")
    shared_l3 = L3Tier(dsn=PG_DSN, table="bench_parallel_l3")
    await shared_l2.connect()
    await shared_l3.connect()
    await shared_l2.clear()
    await shared_l3.clear()

    # Pre-populate L3 with the full corpus
    for key in CORPUS:
        await shared_l3.set(key, {"content": f"finding for {key}", "score": 0.9})

    l3_reads_avoided = 0
    l3_reads_total = 0

    t0 = time.perf_counter()

    # Agents run sequentially (simulating staggered parallel search)
    # Each agent searches the same overlapping key set
    # Agents after the first should hit L2 for already-searched keys
    for agent_idx in range(N_AGENTS):
        l1 = L1Tier(capacity=16)
        h = MemoryHierarchy(tiers=[l1, shared_l2, shared_l3])

        # Each agent reads a subset of the corpus (with overlap between agents)
        keys_to_search = CORPUS[agent_idx * (KEYS_PER_AGENT // 2):
                                 agent_idx * (KEYS_PER_AGENT // 2) + KEYS_PER_AGENT]

        for key in keys_to_search:
            # Check if L2 already has it (cross-agent cache hit)
            l2_has = await shared_l2.exists(key)
            await h.get(key)
            stats.total_reads += 1

            if l2_has:
                stats.l2_hits += 1
                l3_reads_avoided += 1
            else:
                stats.l2_misses += 1
                l3_reads_total += 1

    stats.duration_ms = (time.perf_counter() - t0) * 1000
    stats.extra["l3_reads_avoided"] = l3_reads_avoided
    stats.extra["l3_reads_total"] = l3_reads_total

    await shared_l2.clear()
    await shared_l3.clear()
    await shared_l2.disconnect()
    await shared_l3.disconnect()

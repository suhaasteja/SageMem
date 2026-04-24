"""Research workload — multi-agent fact sharing.

Three agents independently research facts and store them in the hierarchy.
Later reads by other agents hit the cache (via L2 shared tier) instead of
re-fetching from L3.

Measures: L1/L2 hit rates, token savings from cache reuse.
"""

import asyncio
import time

from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy
from sagemem.agent import MemoryAgent
from benchmarks.common import BenchmarkStats, MockLLM

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"

FACTS = [
    ("fact:speed_of_light", "299,792,458 m/s"),
    ("fact:planck_constant", "6.626e-34 J·s"),
    ("fact:avogadro", "6.022e23 mol⁻¹"),
    ("fact:boltzmann", "1.380e-23 J/K"),
    ("fact:gravity", "9.807 m/s²"),
    ("fact:electron_mass", "9.109e-31 kg"),
    ("fact:proton_mass", "1.673e-27 kg"),
    ("fact:pi", "3.14159265358979"),
    ("fact:euler", "2.71828182845905"),
    ("fact:golden_ratio", "1.61803398874989"),
]

N_AGENTS = 3
READS_PER_AGENT = 20


async def run(stats: BenchmarkStats) -> None:
    """Run the research workload and populate stats."""
    shared_l2 = L2Tier(url=REDIS_URL, namespace="bench:research:l2")
    shared_l3 = L3Tier(dsn=PG_DSN, table="bench_research_l3")
    await shared_l2.connect()
    await shared_l3.connect()
    await shared_l2.clear()
    await shared_l3.clear()

    # Seed all facts into L3 (simulates prior research stored globally)
    for key, value in FACTS:
        await shared_l3.set(key, value)

    agents = []
    for i in range(N_AGENTS):
        l1 = L1Tier(capacity=16)
        h = MemoryHierarchy(tiers=[l1, shared_l2, shared_l3])
        agent = MemoryAgent(
            agent_id=f"researcher-{i}",
            hierarchy=h,
            llm=MockLLM(),
            default_write_tier=2,
        )
        agents.append((agent, l1))

    t0 = time.perf_counter()

    # Each agent reads all facts READS_PER_AGENT // len(FACTS) times
    # First read: L1 miss → L2 miss → L3 hit → promote
    # Subsequent reads: L1 hit
    reads_per_key = READS_PER_AGENT // len(FACTS)
    for agent, l1 in agents:
        for _ in range(reads_per_key):
            for key, _ in FACTS:
                value = await agent.recall(key)
                stats.total_reads += 1
                if await l1.exists(key):
                    stats.l1_hits += 1
                else:
                    stats.l1_misses += 1

        # Ask a question using recalled facts (burns tokens)
        context_keys = [k for k, _ in FACTS[:3]]
        await agent.ask("Summarize the key physical constants.", context_keys=context_keys)
        stats.input_tokens += agent.total_tokens

    stats.duration_ms = (time.perf_counter() - t0) * 1000

    await shared_l2.clear()
    await shared_l3.clear()
    await shared_l2.disconnect()
    await shared_l3.disconnect()

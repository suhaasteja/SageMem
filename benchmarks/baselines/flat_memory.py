"""Flat-memory baseline — same workloads with a plain dict per agent.

No tiers, no cross-agent sharing, no coherence. Every agent is an island.
Metrics collected in the same format as tier-aware workloads for comparison.
"""

import time
from benchmarks.common import BenchmarkStats, MockLLM

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
CORPUS_SIZE = 16
KEYS_PER_AGENT = 8
N_PARALLEL_AGENTS = 4


async def run_research_baseline(stats: BenchmarkStats) -> None:
    """Flat-memory version of the research workload."""
    llm = MockLLM()
    t0 = time.perf_counter()

    for _ in range(N_AGENTS):
        store: dict = {}
        # No shared L2/L3 — each agent has to load its own copy
        for key, value in FACTS:
            store[key] = value  # simulates loading from an external DB each time

        reads_per_key = READS_PER_AGENT // len(FACTS)
        for _ in range(reads_per_key):
            for key, _ in FACTS:
                _ = store.get(key)
                stats.total_reads += 1
                # Flat dict: always a "hit" locally, but no L1/L2 distinction
                stats.l1_hits += 1

        # Each agent builds its own context for LLM call — no shared cache
        context = " ".join(v for _, v in FACTS[:3])
        messages = [{"role": "user", "content": f"Known facts:\n[fact:speed_of_light]: {FACTS[0][1]}\n[fact:planck_constant]: {FACTS[1][1]}\n[fact:avogadro]: {FACTS[2][1]}\n\nQuestion: Summarize the key physical constants."}]
        tokens = await llm.count_tokens(messages)
        stats.input_tokens += tokens

    stats.duration_ms = (time.perf_counter() - t0) * 1000


async def run_debate_baseline(stats: BenchmarkStats) -> None:
    """Flat-memory version of the debate workload — no coherence events."""
    beliefs = [
        "belief:origin_of_universe",
        "belief:nature_of_consciousness",
        "belief:optimal_diet",
        "belief:free_will",
        "belief:best_programming_language",
    ]
    store_a: dict = {}
    store_b: dict = {}
    rounds = 5

    t0 = time.perf_counter()

    for round_num in range(rounds):
        for belief in beliefs:
            store_a[belief] = f"A-position-round-{round_num}"
            stats.total_writes += 1

        for belief in beliefs:
            _ = store_b.get(belief)  # B reads its own stale copy — no invalidation
            stats.total_reads += 1
            store_b[belief] = f"B-position-round-{round_num}"
            stats.total_writes += 1

    # No invalidations — agents never know about each other's updates
    stats.coherence_invalidations = 0
    stats.duration_ms = (time.perf_counter() - t0) * 1000


async def run_parallel_search_baseline(stats: BenchmarkStats) -> None:
    """Flat-memory version of the parallel search workload — no cross-agent sharing."""
    corpus = {f"doc:{i:03d}": {"content": f"finding for doc:{i:03d}", "score": 0.9}
              for i in range(CORPUS_SIZE)}

    t0 = time.perf_counter()

    for agent_idx in range(N_PARALLEL_AGENTS):
        store: dict = {}
        start = agent_idx * (KEYS_PER_AGENT // 2)
        keys = list(corpus.keys())[start: start + KEYS_PER_AGENT]

        for key in keys:
            # No shared cache — must always "fetch from DB"
            store[key] = corpus.get(key)
            stats.total_reads += 1
            stats.l2_misses += 1  # would always miss shared cache
            stats.l1_hits += 1   # local store always has it

    stats.extra["l3_reads_avoided"] = 0  # no sharing = no avoidance
    stats.extra["l3_reads_total"] = stats.total_reads
    stats.duration_ms = (time.perf_counter() - t0) * 1000

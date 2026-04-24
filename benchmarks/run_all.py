"""Run all benchmarks and print a comparison table.

Usage:
    uv run python benchmarks/run_all.py

Requires Redis and Postgres running.
"""

import asyncio

from benchmarks.common import BenchmarkStats, print_report
from benchmarks.workloads import research, debate, parallel_search
from benchmarks.baselines import flat_memory


async def main() -> None:
    print("\nSageMem Benchmark Suite")
    print("Running all workloads... (this takes ~10s)\n")

    # -----------------------------------------------------------------------
    # Research workload
    # -----------------------------------------------------------------------
    print("  [1/3] Research workload (multi-agent fact sharing)...")
    s_research = BenchmarkStats(label="research (tier-aware)")
    await research.run(s_research)

    s_research_flat = BenchmarkStats(label="research (flat)")
    await flat_memory.run_research_baseline(s_research_flat)

    # -----------------------------------------------------------------------
    # Debate workload
    # -----------------------------------------------------------------------
    print("  [2/3] Debate workload (concurrent conflicting writes)...")
    s_debate = BenchmarkStats(label="debate (tier-aware)")
    await debate.run(s_debate)

    s_debate_flat = BenchmarkStats(label="debate (flat)")
    await flat_memory.run_debate_baseline(s_debate_flat)

    # -----------------------------------------------------------------------
    # Parallel search workload
    # -----------------------------------------------------------------------
    print("  [3/3] Parallel search workload (cross-agent cache sharing)...")
    s_parallel = BenchmarkStats(label="parallel (tier-aware)")
    await parallel_search.run(s_parallel)

    s_parallel_flat = BenchmarkStats(label="parallel (flat)")
    await flat_memory.run_parallel_search_baseline(s_parallel_flat)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    print("\n=== Results ===")
    print_report([
        s_research, s_research_flat,
        s_debate, s_debate_flat,
        s_parallel, s_parallel_flat,
    ])

    # Key insights
    print("Key insights:")

    if s_research.input_tokens < s_research_flat.input_tokens:
        savings = (1 - s_research.input_tokens / s_research_flat.input_tokens) * 100
        print(f"  Research  : {savings:.1f}% fewer tokens (tier-aware context assembly)")
    else:
        print(f"  Research  : tokens comparable — both inject same facts as context")

    inv = s_debate.coherence_invalidations
    print(f"  Debate    : {inv} invalidation events fired (flat baseline: 0)")
    print(f"            → tier-aware agents converge on new values; flat agents see stale data")

    avoided = s_parallel.extra.get('l3_reads_avoided', 0)
    total_p = s_parallel.total_reads
    if total_p > 0:
        pct = avoided / total_p * 100
        print(f"  Parallel  : {avoided}/{total_p} reads ({pct:.0f}%) served from L2 cross-agent cache")
        print(f"            → flat baseline: 0/{total_p} (no sharing between agents)")
    print()


if __name__ == "__main__":
    asyncio.run(main())

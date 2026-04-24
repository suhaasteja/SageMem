"""Shared utilities for benchmarks: mock LLM, metrics collection, reporting."""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class BenchmarkStats:
    """Collected metrics for one benchmark run."""

    label: str
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    total_reads: int = 0
    total_writes: int = 0
    input_tokens: int = 0
    coherence_invalidations: int = 0
    l3_conflict_retries: int = 0
    duration_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        hits = self.l1_hits + self.l2_hits + self.l3_hits
        return hits / self.total_reads if self.total_reads > 0 else 0.0


class MockLLM:
    """Fake LLM that returns deterministic responses without API calls.

    Used in benchmarks to isolate memory-layer performance from network latency.
    Token counts are based on simple word counting to simulate realistic usage.
    """

    def __init__(self, response_template: str = "Answer: {question}") -> None:
        self.response_template = response_template
        self._tokens_counted = 0

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        await asyncio.sleep(0.001)  # simulate minimal network latency
        content = messages[-1]["content"] if messages else ""
        return f"[mock response to: {content[:60]}]"

    async def count_tokens(self, messages: list[dict]) -> int:
        total = sum(len(m.get("content", "").split()) * 4 // 3 for m in messages)
        self._tokens_counted += total
        return total


def print_report(stats_list: list[BenchmarkStats]) -> None:
    """Print a comparison table of benchmark results."""
    print(f"\n{'─' * 75}")
    print(f"{'Workload':<22} {'L1 hit%':>8} {'Overall%':>9} {'Tokens':>8} "
          f"{'Invalidations':>14} {'Duration':>10}")
    print(f"{'─' * 75}")
    for s in stats_list:
        print(
            f"{s.label:<22} "
            f"{s.l1_hit_rate * 100:>7.1f}% "
            f"{s.overall_hit_rate * 100:>8.1f}% "
            f"{s.input_tokens:>8} "
            f"{s.coherence_invalidations:>14} "
            f"{s.duration_ms:>9.1f}ms"
        )
    print(f"{'─' * 75}\n")

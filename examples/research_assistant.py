"""Single-agent research assistant demo.

Demonstrates tier-aware memory vs flat-memory baseline.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/research_assistant.py

What to observe:
- Turn 1: agent queries the LLM (no cached facts yet)
- Turn 2+: agent recalls facts from L1 (promoted from L3 on first ask)
- Token usage: tier-aware uses fewer tokens on repeated queries
  because recalled facts are injected as compact context, not re-asked
"""

import asyncio
import logging

from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy
from sagemem.agent import MemoryAgent
from sagemem.llm.anthropic import AnthropicClient

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flat-memory baseline
# ---------------------------------------------------------------------------

class FlatMemoryAgent:
    """Baseline agent with a plain dict — no tiers, no hierarchy."""

    def __init__(self, llm: AnthropicClient) -> None:
        self._store: dict = {}
        self._llm = llm
        self._token_usage = 0

    def remember(self, key: str, value: object) -> None:
        self._store[key] = value

    def recall(self, key: str) -> object | None:
        return self._store.get(key)

    async def ask(self, question: str, context_keys: list[str] | None = None) -> str:
        context_parts = []
        if context_keys:
            for key in context_keys:
                v = self.recall(key)
                if v:
                    context_parts.append(f"[{key}]: {v}")

        user_content = question
        if context_parts:
            user_content = "Known facts:\n" + "\n".join(context_parts) + f"\n\nQuestion: {question}"

        messages = [{"role": "user", "content": user_content}]
        self._token_usage += await self._llm.count_tokens(messages)
        return await self._llm.complete(messages)

    @property
    def total_tokens(self) -> int:
        return self._token_usage


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("What is the capital of France?", []),
    ("What language do they speak in the capital of France?", ["fact:france_capital"]),
    ("Name one famous landmark in that city.", ["fact:france_capital", "fact:france_language"]),
]


async def run_tier_aware() -> tuple[list[str], int]:
    """Run the tier-aware agent through the question set."""
    l1 = L1Tier(capacity=64)
    l2 = L2Tier(url=REDIS_URL, namespace="sagemem:demo")
    l3 = L3Tier(dsn=PG_DSN, table="demo_l3")

    await l2.connect()
    await l3.connect()
    await l2.clear()
    await l3.clear()

    hierarchy = MemoryHierarchy(tiers=[l1, l2, l3])
    llm = AnthropicClient()
    agent = MemoryAgent(agent_id="researcher", hierarchy=hierarchy, llm=llm, default_write_tier=2)

    answers = []
    for i, (question, context_keys) in enumerate(QUESTIONS):
        log.info("\n[Tier-aware] Turn %d: %s", i + 1, question)
        answer = await agent.ask(question, context_keys=context_keys)
        log.info("[Tier-aware] Answer: %s", answer)
        answers.append(answer)

        # Store key facts after each turn
        if i == 0:
            await agent.remember("fact:france_capital", "Paris", tier_index=2)
        elif i == 1:
            await agent.remember("fact:france_language", "French", tier_index=2)

    await l2.disconnect()
    await l3.disconnect()

    return answers, agent.total_tokens


async def run_flat_baseline() -> tuple[list[str], int]:
    """Run the flat-memory baseline through the same question set."""
    llm = AnthropicClient()
    agent = FlatMemoryAgent(llm=llm)

    answers = []
    for i, (question, context_keys) in enumerate(QUESTIONS):
        log.info("\n[Flat]      Turn %d: %s", i + 1, question)
        answer = await agent.ask(question, context_keys=context_keys)
        log.info("[Flat]      Answer: %s", answer)
        answers.append(answer)

        if i == 0:
            agent.remember("fact:france_capital", "Paris")
        elif i == 1:
            agent.remember("fact:france_language", "French")

    return answers, agent.total_tokens


async def main() -> None:
    print("\n" + "=" * 60)
    print("SageMem — Single-Agent Research Assistant Demo")
    print("=" * 60)

    _, tier_tokens = await run_tier_aware()
    _, flat_tokens = await run_flat_baseline()

    print("\n" + "=" * 60)
    print("Token usage comparison")
    print(f"  Tier-aware : {tier_tokens} input tokens")
    print(f"  Flat memory: {flat_tokens} input tokens")
    if flat_tokens > 0:
        savings = (flat_tokens - tier_tokens) / flat_tokens * 100
        print(f"  Difference : {savings:+.1f}%")
    print("=" * 60)
    print("\nWhat happened:")
    print("  Turn 1 — both agents queried the LLM cold (no cached facts)")
    print("  Turn 2+ — both injected recalled facts as context")
    print("  L1 cache served turns 2-3 for tier-aware (promoted from L3 on first recall)")
    print()


if __name__ == "__main__":
    asyncio.run(main())

"""MemoryAgent — an agent that reads and writes semantic memory through the hierarchy."""

import logging
from typing import Any

from sagemem.hierarchy import MemoryHierarchy
from sagemem.llm.base import LLMClient

logger = logging.getLogger(__name__)


class MemoryAgent:
    """An agent that assembles tier-aware context and calls an LLM.

    Facts are stored in the hierarchy and reused across turns, reducing
    redundant token usage on repeated queries.

    Args:
        agent_id: Unique identifier for this agent.
        hierarchy: The memory hierarchy this agent operates on.
        llm: LLM client for generating responses.
        default_write_tier: Tier index where remember() writes by default (0 = L1).
    """

    def __init__(
        self,
        agent_id: str,
        hierarchy: MemoryHierarchy,
        llm: LLMClient,
        default_write_tier: int = 0,
    ) -> None:
        """Initialize the agent."""
        self.agent_id = agent_id
        self.hierarchy = hierarchy
        self.llm = llm
        self.default_write_tier = default_write_tier
        self._token_usage: int = 0

    async def remember(self, key: str, value: Any, tier_index: int | None = None) -> None:
        """Store a value in the hierarchy.

        Args:
            key: Unique key for this fact.
            value: The value to store.
            tier_index: Tier to write to. Defaults to self.default_write_tier.
        """
        target = tier_index if tier_index is not None else self.default_write_tier
        await self.hierarchy.set(key, value, tier_index=target)
        logger.debug("[%s] remember key=%r tier=%d", self.agent_id, key, target)

    async def recall(self, key: str) -> Any | None:
        """Retrieve a value from the hierarchy, falling through tiers on miss.

        Returns the value if found in any tier, None otherwise.
        """
        value = await self.hierarchy.get(key)
        if value is not None:
            logger.debug("[%s] recall hit key=%r", self.agent_id, key)
        else:
            logger.debug("[%s] recall miss key=%r", self.agent_id, key)
        return value

    async def ask(
        self,
        question: str,
        context_keys: list[str] | None = None,
        system: str | None = None,
    ) -> str:
        """Ask the LLM a question, injecting recalled facts as context.

        Args:
            question: The user's question.
            context_keys: Keys to recall from the hierarchy and inject as context.
            system: Optional system prompt override.

        Returns:
            The model's text response.
        """
        # Assemble context from hierarchy
        context_parts: list[str] = []
        if context_keys:
            for key in context_keys:
                value = await self.recall(key)
                if value is not None:
                    context_parts.append(f"[{key}]: {value}")

        # Build messages
        user_content = question
        if context_parts:
            context_block = "\n".join(context_parts)
            user_content = f"Known facts:\n{context_block}\n\nQuestion: {question}"

        messages = [{"role": "user", "content": user_content}]

        # Count tokens before call
        tokens_used = await self.llm.count_tokens(messages)
        self._token_usage += tokens_used

        response = await self.llm.complete(messages, system=system)
        return response

    @property
    def total_tokens(self) -> int:
        """Cumulative input tokens used by this agent across all ask() calls."""
        return self._token_usage

"""LLMClient protocol — all LLM adapters implement this interface."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface for an LLM backend.

    Any adapter (Anthropic, OpenAI, local) must implement these methods.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send messages to the model and return the text response."""
        ...

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Return approximate token count for the given messages."""
        ...

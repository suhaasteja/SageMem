"""Anthropic adapter implementing the LLMClient protocol."""

from typing import Any

import anthropic


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API.

    Args:
        model: Model ID to use (defaults to claude-haiku-4-5-20251001 for speed/cost).
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        """Initialize with model name and optional API key."""
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send messages to the model and return the text response."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Return approximate token count for the given messages."""
        response = await self._client.messages.count_tokens(
            model=self.model,
            messages=messages,
        )
        return response.input_tokens

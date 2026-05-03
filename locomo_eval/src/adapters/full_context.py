"""Full-context baseline: stuff the entire conversation into the prompt."""
import time
from collections import defaultdict

import tiktoken

from .base import AddStats, MemoryAdapter, SearchResult

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


class FullContextAdapter(MemoryAdapter):
    """Appends every turn to an in-memory list; search returns everything."""

    name = "full_context"

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = defaultdict(list)

    def reset(self, conversation_id: str) -> None:
        self._store[conversation_id] = []

    def add_session(
        self,
        conversation_id: str,
        session_id: str,
        timestamp: str,
        turns: list[dict],
    ) -> AddStats:
        t0 = time.perf_counter()
        lines = [f"[{timestamp}]"]
        for turn in turns:
            lines.append(f"{turn['speaker']}: {turn['text']}")
        self._store[conversation_id].extend(lines)
        return AddStats(latency_ms=(time.perf_counter() - t0) * 1000)

    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 30,
    ) -> SearchResult:
        t0 = time.perf_counter()
        context = "\n".join(self._store[conversation_id])
        return SearchResult(
            context=context,
            retrieved_tokens=_count_tokens(context),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

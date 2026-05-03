"""SageMem adapter — wires the LoCoMo eval harness to the sagemem memory system."""
import sys
import time
from pathlib import Path

import tiktoken

from .base import AddStats, MemoryAdapter, SearchResult

# Make the parent sagemem package importable from this sub-project
_SAGEMEM_SRC = Path(__file__).parents[4] / "src"
if str(_SAGEMEM_SRC) not in sys.path:
    sys.path.insert(0, str(_SAGEMEM_SRC))

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


class SageMemAdapter(MemoryAdapter):
    """
    Adapter for the sagemem memory hierarchy.

    Ingestion: turns are stored in L1/L2/L3 using the MemoryHierarchy.
    Retrieval: DRAM semantic search returns top-k chunks as context.

    NOTE: This adapter uses synchronous wrappers around async sagemem calls.
    If sagemem grows a sync API, prefer that. For now we use asyncio.run().
    """

    name = "my_system"

    def __init__(self) -> None:
        # Lazy import so the harness can run without sagemem deps installed
        # (e.g. when testing baselines only).
        try:
            import asyncio
            from sagemem.hierarchy import MemoryHierarchy
            from sagemem.tiers.l1 import L1Cache
            from sagemem.tiers.l2 import L2Cache
            from sagemem.tiers.l3 import L3Store
            from sagemem.tiers.dram import DRAMStore
            self._asyncio = asyncio
            self._MemoryHierarchy = MemoryHierarchy
            self._L1 = L1Cache
            self._available = True
        except ImportError as e:
            print(f"[SageMemAdapter] sagemem not importable: {e}. Falling back to stub.")
            self._available = False

        self._hierarchies: dict[str, object] = {}

    def _get_hierarchy(self, conversation_id: str):
        if conversation_id not in self._hierarchies:
            self._hierarchies[conversation_id] = self._MemoryHierarchy()
        return self._hierarchies[conversation_id]

    def reset(self, conversation_id: str) -> None:
        if conversation_id in self._hierarchies:
            h = self._hierarchies.pop(conversation_id)
            try:
                self._asyncio.run(h.clear())
            except Exception:
                pass

    def add_session(
        self,
        conversation_id: str,
        session_id: str,
        timestamp: str,
        turns: list[dict],
    ) -> AddStats:
        if not self._available:
            return AddStats()

        t0 = time.perf_counter()
        h = self._get_hierarchy(conversation_id)

        async def _ingest():
            for turn in turns:
                key = f"{conversation_id}:{session_id}:{turn['dia_id']}"
                value = {
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "timestamp": timestamp,
                    "session": session_id,
                }
                await h.set(key, value)

        self._asyncio.run(_ingest())
        return AddStats(latency_ms=(time.perf_counter() - t0) * 1000)

    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 30,
    ) -> SearchResult:
        if not self._available:
            return SearchResult(context="", retrieved_tokens=0, latency_ms=0.0)

        t0 = time.perf_counter()
        h = self._get_hierarchy(conversation_id)

        async def _search():
            return await h.semantic_search(query, top_k=top_k)

        results = self._asyncio.run(_search())
        context = "\n\n".join(
            f"{r['speaker']}: {r['text']}" for r in results if isinstance(r, dict)
        )
        return SearchResult(
            context=context,
            retrieved_tokens=_count_tokens(context),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

"""SageMem adapter — wires the LoCoMo eval harness to the sagemem memory system."""
import os
import sys
import time
from pathlib import Path

from .base import AddStats, MemoryAdapter, SearchResult

# Make the parent sagemem package importable from this sub-project
_SAGEMEM_SRC = Path(__file__).parents[3] / "src"
if str(_SAGEMEM_SRC) not in sys.path:
    sys.path.insert(0, str(_SAGEMEM_SRC))

_PG_DSN = os.environ.get("SAGEMEM_PG_DSN", "postgresql://localhost/sagemem_test")


def _count_tokens(text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _make_embedder():
    """Return an async embedding callable using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    async def _embed(text: str) -> list[float]:
        return _model.encode(text, show_progress_bar=False).tolist()

    return _embed


class SageMemAdapter(MemoryAdapter):
    """
    Adapter for the sagemem memory hierarchy.

    Ingestion: turns are stored in L1 + DRAM using MemoryHierarchy.
    Retrieval: DRAM semantic search returns top-k chunks as context.

    Requires Postgres with pgvector (SAGEMEM_PG_DSN env var, default:
    postgresql://localhost/sagemem_test).

    NOTE: Uses asyncio.run() wrappers because the MemoryAdapter interface
    is synchronous.
    """

    name = "my_system"

    def __init__(self) -> None:
        try:
            import asyncio
            from sagemem.hierarchy import MemoryHierarchy
            from sagemem.tiers.l1 import L1Tier
            from sagemem.tiers.dram import DRAMTier
            self._asyncio = asyncio
            self._MemoryHierarchy = MemoryHierarchy
            self._L1Tier = L1Tier
            self._DRAMTier = DRAMTier
            self._embedder = _make_embedder()
            self._available = True
        except ImportError as e:
            print(f"[SageMemAdapter] sagemem not importable: {e}. Falling back to stub.")
            self._available = False

        self._hierarchies: dict[str, object] = {}

    def _get_hierarchy(self, conversation_id: str):
        if conversation_id not in self._hierarchies:
            safe_id = conversation_id.replace("-", "_").replace(":", "_")
            async def _make():
                l1 = self._L1Tier(capacity=512)
                dram = self._DRAMTier(
                    dsn=_PG_DSN,
                    table=f"locomo_{safe_id}",
                    embedding_dim=384,
                    embedder=self._embedder,
                )
                await dram.connect()
                return self._MemoryHierarchy(tiers=[l1, dram])
            self._hierarchies[conversation_id] = self._asyncio.run(_make())
        return self._hierarchies[conversation_id]

    def reset(self, conversation_id: str) -> None:
        if conversation_id in self._hierarchies:
            h = self._hierarchies.pop(conversation_id)
            try:
                async def _teardown():
                    await h.clear()
                    # Disconnect DRAM pool (last tier)
                    if hasattr(h.tiers[-1], "disconnect"):
                        await h.tiers[-1].disconnect()
                self._asyncio.run(_teardown())
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
                # Write to DRAM (tier_index=1) so embeddings are computed
                await h.set(key, value, tier_index=1)

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

        # Each result is the stored value dict: {"speaker": ..., "text": ..., ...}
        results = self._asyncio.run(_search())
        context = "\n\n".join(
            f"{r['speaker']}: {r['text']}" for r in results if isinstance(r, dict)
        )
        return SearchResult(
            context=context,
            retrieved_tokens=_count_tokens(context),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

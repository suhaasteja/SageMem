"""Naive RAG baseline: chunk turns, embed with text-embedding-3-small, FAISS top-k."""
import time
from collections import defaultdict

import numpy as np
import tiktoken
from openai import OpenAI

from .base import AddStats, MemoryAdapter, SearchResult

_enc = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 500


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _chunk_text(text: str, max_tokens: int = CHUNK_TOKENS) -> list[str]:
    tokens = _enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(_enc.decode(tokens[i : i + max_tokens]))
    return chunks or [text]


class NaiveRAGAdapter(MemoryAdapter):
    """Chunks all turns, embeds them, stores in a per-conversation FAISS index."""

    name = "naive_rag"

    def __init__(self, client: OpenAI, embed_model: str = "text-embedding-3-small") -> None:
        self._client = client
        self._embed_model = embed_model
        # conversation_id -> list of (chunk_text, embedding_vector)
        self._stores: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)

    def reset(self, conversation_id: str) -> None:
        self._stores[conversation_id] = []

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        response = self._client.embeddings.create(model=self._embed_model, input=texts)
        return [np.array(d.embedding, dtype=np.float32) for d in response.data]

    def add_session(
        self,
        conversation_id: str,
        session_id: str,
        timestamp: str,
        turns: list[dict],
    ) -> AddStats:
        t0 = time.perf_counter()
        lines = [f"[{timestamp}] {t['speaker']}: {t['text']}" for t in turns]
        combined = "\n".join(lines)
        chunks = _chunk_text(combined)
        embeddings = self._embed(chunks)
        for chunk, emb in zip(chunks, embeddings):
            self._stores[conversation_id].append((chunk, emb))
        return AddStats(latency_ms=(time.perf_counter() - t0) * 1000)

    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 30,
    ) -> SearchResult:
        t0 = time.perf_counter()
        store = self._stores[conversation_id]

        if not store:
            return SearchResult(context="", retrieved_tokens=0, latency_ms=0.0)

        query_emb = self._embed([query])[0]
        chunks, vecs = zip(*store)
        matrix = np.stack(vecs)

        # cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_emb)
        scores = (matrix @ query_emb) / (norms.squeeze() * query_norm + 1e-9)

        top_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
        top_chunks = [chunks[i] for i in sorted(top_indices)]  # preserve chronological order
        context = "\n\n".join(top_chunks)

        return SearchResult(
            context=context,
            retrieved_tokens=_count_tokens(context),
            latency_ms=(time.perf_counter() - t0) * 1000,
            chunk_ids=[str(i) for i in top_indices],
        )

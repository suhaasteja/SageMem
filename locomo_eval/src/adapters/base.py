"""Memory adapter ABC — all adapters implement this contract."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AddStats:
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class SearchResult:
    context: str
    retrieved_tokens: int
    latency_ms: float
    chunk_ids: list[str] = field(default_factory=list)


class MemoryAdapter(ABC):
    name: str

    @abstractmethod
    def reset(self, conversation_id: str) -> None:
        """Clear all memory for this conversation. Called before each run."""

    @abstractmethod
    def add_session(
        self,
        conversation_id: str,
        session_id: str,
        timestamp: str,
        turns: list[dict],
    ) -> AddStats:
        """Ingest one session in chronological order."""

    @abstractmethod
    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 30,
    ) -> SearchResult:
        """Return retrieved context and metadata."""

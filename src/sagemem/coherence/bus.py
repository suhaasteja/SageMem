"""CoherenceBus — transport layer for MESI invalidation messages.

Stage 0: stubbed. Stage 4 wires this to Redis pub/sub.
"""

from dataclasses import dataclass


@dataclass
class InvalidateMessage:
    """Broadcast when a writer modifies a Shared or Exclusive entry."""

    key: str
    writer_id: str
    new_version: int


class CoherenceBus:
    """Stub coherence bus. Publishes and receives invalidation events."""

    async def publish_invalidate(self, msg: InvalidateMessage) -> None:
        """Broadcast an invalidation message to all subscribed agents."""
        raise NotImplementedError("CoherenceBus is a stub — wired in Stage 4.")

    async def subscribe(self, agent_id: str, callback: object) -> None:
        """Subscribe agent_id to receive invalidation messages via callback."""
        raise NotImplementedError("CoherenceBus is a stub — wired in Stage 4.")

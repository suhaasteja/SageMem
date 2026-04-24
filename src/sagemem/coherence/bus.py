"""CoherenceBus — Redis pub/sub transport for MESI invalidation messages."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

CHANNEL = "sagemem:coherence"


@dataclass
class InvalidateMessage:
    """Broadcast when a writer modifies a Shared or Exclusive entry."""

    key: str
    writer_id: str
    new_version: int

    def to_json(self) -> str:
        """Serialize to JSON string for pub/sub transport."""
        return json.dumps({
            "key": self.key,
            "writer_id": self.writer_id,
            "new_version": self.new_version,
        })

    @classmethod
    def from_json(cls, raw: str) -> "InvalidateMessage":
        """Deserialize from JSON string."""
        data = json.loads(raw)
        return cls(
            key=data["key"],
            writer_id=data["writer_id"],
            new_version=data["new_version"],
        )


Callback = Callable[[InvalidateMessage], Awaitable[None]]


class CoherenceBus:
    """Redis pub/sub coherence bus.

    Agents publish invalidation messages when they write to a Shared/Exclusive
    key. All subscribed agents receive these messages and mark their local
    cache entry as Invalid.

    Args:
        url: Redis connection URL.
        channel: Pub/sub channel name. All agents in a system must use the same channel.
    """

    def __init__(self, url: str = "redis://localhost:6379", channel: str = CHANNEL) -> None:
        """Initialize with Redis URL and channel name."""
        self.url = url
        self.channel = channel
        self._pub: aioredis.Redis | None = None
        self._sub: aioredis.Redis | None = None
        self._listener_task: asyncio.Task | None = None
        self._callbacks: dict[str, Callback] = {}  # agent_id -> callback

    async def connect(self) -> None:
        """Open publisher and subscriber Redis connections."""
        self._pub = aioredis.from_url(self.url, decode_responses=True)
        self._sub = aioredis.from_url(self.url, decode_responses=True)

    async def disconnect(self) -> None:
        """Stop the listener and close connections."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        if self._pub:
            await self._pub.aclose()
        if self._sub:
            await self._sub.aclose()

    async def publish_invalidate(self, msg: InvalidateMessage) -> None:
        """Broadcast an invalidation message to all subscribed agents."""
        if self._pub is None:
            raise RuntimeError("CoherenceBus not connected.")
        await self._pub.publish(self.channel, msg.to_json())
        logger.debug("[bus] published invalidate key=%r writer=%r v=%d",
                     msg.key, msg.writer_id, msg.new_version)

    async def subscribe(self, agent_id: str, callback: Callback) -> None:
        """Register agent_id to receive invalidation messages via callback.

        Starts the background listener if not already running.
        """
        self._callbacks[agent_id] = callback
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Background task: receive pub/sub messages and dispatch to callbacks."""
        if self._sub is None:
            raise RuntimeError("CoherenceBus not connected.")
        pubsub = self._sub.pubsub()
        await pubsub.subscribe(self.channel)
        logger.debug("[bus] listener started on channel=%r", self.channel)
        try:
            async for raw_msg in pubsub.listen():
                if raw_msg["type"] != "message":
                    continue
                try:
                    msg = InvalidateMessage.from_json(raw_msg["data"])
                except (KeyError, json.JSONDecodeError):
                    logger.warning("[bus] malformed message: %r", raw_msg["data"])
                    continue
                for agent_id, cb in list(self._callbacks.items()):
                    try:
                        await cb(msg)
                    except Exception:
                        logger.exception("[bus] callback error for agent=%r", agent_id)
        except asyncio.CancelledError:
            await pubsub.unsubscribe(self.channel)
            raise

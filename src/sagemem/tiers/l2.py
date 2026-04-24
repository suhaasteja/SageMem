"""L2 tier — per-cluster Redis-backed cache."""

import json
from typing import Any

import redis.asyncio as aioredis

from sagemem.tiers.base import Tier


class L2Tier(Tier):
    """Redis-backed cache. Shared within a cluster of agents.

    Args:
        url: Redis connection URL (e.g. 'redis://localhost:6379').
        namespace: Key prefix to isolate this cluster's data.
        ttl: Optional TTL in seconds for all entries. None = no expiry.
    """

    def __init__(self, url: str = "redis://localhost:6379", namespace: str = "sagemem:l2", ttl: int | None = None) -> None:
        """Initialize with connection URL and optional namespace/TTL."""
        self.url = url
        self.namespace = namespace
        self.ttl = ttl
        self._client: aioredis.Redis | None = None

    def _client_or_raise(self) -> aioredis.Redis:
        """Return the Redis client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("L2Tier not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Open the Redis connection."""
        self._client = aioredis.from_url(self.url, decode_responses=True)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _key(self, key: str) -> str:
        """Prefix key with the namespace."""
        return f"{self.namespace}:{key}"

    async def get(self, key: str) -> Any | None:
        """Return deserialized value for key, or None if not present."""
        raw = await self._client_or_raise().get(self._key(key))
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any) -> None:
        """Serialize and store value under key, with optional TTL."""
        client = self._client_or_raise()
        serialized = json.dumps(value)
        if self.ttl is not None:
            await client.setex(self._key(key), self.ttl, serialized)
        else:
            await client.set(self._key(key), serialized)

    async def delete(self, key: str) -> None:
        """Remove key from Redis."""
        await self._client_or_raise().delete(self._key(key))

    async def clear(self) -> None:
        """Remove all keys under this namespace."""
        client = self._client_or_raise()
        pattern = f"{self.namespace}:*"
        keys = await client.keys(pattern)
        if keys:
            await client.delete(*keys)

    async def exists(self, key: str) -> bool:
        """Return True if key exists in Redis."""
        return bool(await self._client_or_raise().exists(self._key(key)))

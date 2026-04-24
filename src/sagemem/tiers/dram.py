"""DRAM tier — Postgres + pgvector for semantic (embedding-based) retrieval."""

import json
from typing import Any

import asyncpg

from sagemem.tiers.base import Tier


class DRAMTier(Tier):
    """Postgres + pgvector tier for semantic memory.

    Supports exact key lookup (like other tiers) and semantic search by embedding.

    Args:
        dsn: Postgres connection string.
        table: Table name to use.
        embedding_dim: Dimension of the embedding vectors.
        embedder: Callable that takes a string and returns a list[float].
                  If None, semantic search is unavailable.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "sagemem_dram",
        embedding_dim: int = 384,
        embedder: Any | None = None,
    ) -> None:
        """Initialize with DSN, table name, embedding dimension, and optional embedder."""
        self.dsn = dsn
        self.table = table
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool, enable pgvector, and ensure the table exists."""
        self._pool = await asyncpg.create_pool(self.dsn)
        await self._migrate()

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _pool_or_raise(self) -> asyncpg.Pool:
        """Return the pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError("DRAMTier not connected. Call connect() first.")
        return self._pool

    async def _migrate(self) -> None:
        """Enable pgvector extension and create the storage table if needed."""
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    key       TEXT PRIMARY KEY,
                    value     JSONB NOT NULL,
                    embedding vector({self.embedding_dim})
                )
            """)

    async def get(self, key: str) -> Any | None:
        """Return the stored value for key, or None if not present."""
        row = await self._pool_or_raise().fetchrow(
            f"SELECT value FROM {self.table} WHERE key = $1", key
        )
        if row is None:
            return None
        return json.loads(row["value"])

    async def set(self, key: str, value: Any) -> None:
        """Store value under key. Computes and stores embedding if embedder is set."""
        serialized = json.dumps(value)
        embedding = None
        if self.embedder is not None:
            text = value if isinstance(value, str) else json.dumps(value)
            embedding = await self.embedder(text)

        if embedding is not None:
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await self._pool_or_raise().execute(f"""
                INSERT INTO {self.table} (key, value, embedding)
                VALUES ($1, $2::jsonb, $3::vector)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        embedding = EXCLUDED.embedding
            """, key, serialized, embedding_str)
        else:
            await self._pool_or_raise().execute(f"""
                INSERT INTO {self.table} (key, value)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value
            """, key, serialized)

    async def delete(self, key: str) -> None:
        """Remove key from the table."""
        await self._pool_or_raise().execute(
            f"DELETE FROM {self.table} WHERE key = $1", key
        )

    async def clear(self) -> None:
        """Remove all rows from the table."""
        await self._pool_or_raise().execute(f"TRUNCATE {self.table}")

    async def exists(self, key: str) -> bool:
        """Return True if key exists in the table."""
        row = await self._pool_or_raise().fetchrow(
            f"SELECT 1 FROM {self.table} WHERE key = $1", key
        )
        return row is not None

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k entries by cosine similarity to query_embedding.

        Requires pgvector and that entries were stored with embeddings.
        """
        if not query_embedding:
            raise ValueError("query_embedding must be a non-empty list of floats.")
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        rows = await self._pool_or_raise().fetch(f"""
            SELECT key, value, embedding <=> $1::vector AS distance
            FROM {self.table}
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT $2
        """, embedding_str, top_k)
        return [
            {"key": row["key"], "value": json.loads(row["value"]), "distance": row["distance"]}
            for row in rows
        ]

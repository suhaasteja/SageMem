"""L3 tier — global Postgres JSONB store with optimistic concurrency."""

import json
from typing import Any

import asyncpg

from sagemem.tiers.base import Tier


class VersionConflictError(Exception):
    """Raised when an L3 write fails due to a version mismatch."""


class L3Tier(Tier):
    """Postgres-backed global store. Uses JSONB and a version column for CAS.

    Args:
        dsn: Postgres connection string.
        table: Table name to use for storage.
    """

    def __init__(self, dsn: str, table: str = "sagemem_l3") -> None:
        """Initialize with a Postgres DSN and table name."""
        self.dsn = dsn
        self.table = table
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool and ensure the table exists."""
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
            raise RuntimeError("L3Tier not connected. Call connect() first.")
        return self._pool

    async def _migrate(self) -> None:
        """Create the storage table if it does not exist."""
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    key     TEXT PRIMARY KEY,
                    value   JSONB NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
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

    async def get_versioned(self, key: str) -> tuple[Any, int] | None:
        """Return (value, version) for key, or None if not present."""
        row = await self._pool_or_raise().fetchrow(
            f"SELECT value, version FROM {self.table} WHERE key = $1", key
        )
        if row is None:
            return None
        return json.loads(row["value"]), row["version"]

    async def set(self, key: str, value: Any) -> None:
        """Upsert value unconditionally, incrementing the version."""
        serialized = json.dumps(value)
        await self._pool_or_raise().execute(f"""
            INSERT INTO {self.table} (key, value, version)
            VALUES ($1, $2::jsonb, 0)
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value,
                    version = {self.table}.version + 1
        """, key, serialized)

    async def set_versioned(self, key: str, value: Any, expected_version: int) -> None:
        """CAS write — only succeeds if current version matches expected_version.

        Raises VersionConflictError on mismatch.
        """
        serialized = json.dumps(value)
        result = await self._pool_or_raise().execute(f"""
            UPDATE {self.table}
               SET value = $1::jsonb,
                   version = version + 1
             WHERE key = $2 AND version = $3
        """, serialized, key, expected_version)
        # result is a tag like "UPDATE 1" or "UPDATE 0"
        if result == "UPDATE 0":
            # Also try insert if row doesn't exist yet
            try:
                await self._pool_or_raise().execute(f"""
                    INSERT INTO {self.table} (key, value, version)
                    VALUES ($1, $2::jsonb, 0)
                """, key, serialized)
            except asyncpg.UniqueViolationError:
                raise VersionConflictError(
                    f"Version conflict on key {key!r}: expected {expected_version}"
                )

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

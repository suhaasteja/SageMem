"""Abstract base class for all memory tiers."""

from abc import ABC, abstractmethod
from typing import Any


class Tier(ABC):
    """Abstract memory tier. All tiers implement async get/set/delete/clear."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Return the value for key, or None if not present."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Store value under key."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove key from this tier."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries from this tier."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if key is present in this tier."""
        ...

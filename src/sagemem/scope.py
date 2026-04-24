"""Capacity-as-capability: scoped access to the memory hierarchy.

AgentCapability declares which tiers an agent can access and its token budget.
ScopedHierarchy wraps MemoryHierarchy and enforces the capability boundary —
an agent restricted to L1+L2 physically cannot see L3 or DRAM data.
"""

from dataclasses import dataclass, field
from typing import Any

from sagemem.hierarchy import MemoryHierarchy


class ScopeViolationError(Exception):
    """Raised when an agent attempts to access a tier outside its capability."""


@dataclass
class AgentCapability:
    """Declares the memory tiers and budgets available to an agent.

    Args:
        tiers: Set of tier indices the agent can access (0=L1, 1=L2, 2=L3, 3=DRAM).
        l1_budget_tokens: Maximum tokens this agent may store in L1.
        l2_read: Whether the agent may read from L2.
        l2_write: Whether the agent may write to L2.
        l3_read: Whether the agent may read from L3.
        l3_write: Whether the agent may write to L3.
        dram_read: Whether the agent may read from DRAM.
        dram_write: Whether the agent may write to DRAM.
    """

    tiers: set[int] = field(default_factory=lambda: {0, 1, 2, 3})
    l1_budget_tokens: int = 8000
    l2_read: bool = True
    l2_write: bool = True
    l3_read: bool = True
    l3_write: bool = True
    dram_read: bool = True
    dram_write: bool = True

    def can_read(self, tier_index: int) -> bool:
        """Return True if this agent may read from the given tier."""
        if tier_index not in self.tiers:
            return False
        return {
            0: True,       # L1 always readable if in tiers
            1: self.l2_read,
            2: self.l3_read,
            3: self.dram_read,
        }.get(tier_index, False)

    def can_write(self, tier_index: int) -> bool:
        """Return True if this agent may write to the given tier."""
        if tier_index not in self.tiers:
            return False
        return {
            0: True,        # L1 always writable if in tiers
            1: self.l2_write,
            2: self.l3_write,
            3: self.dram_write,
        }.get(tier_index, False)


class ScopedHierarchy:
    """A capability-enforced view over a MemoryHierarchy.

    Reads only fall through tiers the agent is permitted to access.
    Writes to out-of-scope tiers raise ScopeViolationError immediately.

    Args:
        hierarchy: The underlying MemoryHierarchy.
        capability: The agent's declared capability.
        agent_id: Agent identifier (used in error messages).
    """

    def __init__(
        self,
        hierarchy: MemoryHierarchy,
        capability: AgentCapability,
        agent_id: str,
    ) -> None:
        """Initialize with a hierarchy, capability, and agent identity."""
        self.hierarchy = hierarchy
        self.capability = capability
        self.agent_id = agent_id

    def _readable_tiers(self) -> list[int]:
        """Return indices of tiers this agent may read from, in order."""
        return [
            i for i in range(len(self.hierarchy.tiers))
            if self.capability.can_read(i)
        ]

    async def get(self, key: str) -> Any | None:
        """Read key, falling through only permitted tiers.

        Returns the value if found in any accessible tier, None otherwise.
        Does NOT promote to tiers the agent cannot write to.
        """
        readable = self._readable_tiers()
        for i in readable:
            tier = self.hierarchy.tiers[i]
            value = await tier.get(key)
            if value is not None:
                # Promote to faster readable+writable tiers
                for j in readable:
                    if j >= i:
                        break
                    if self.capability.can_write(j):
                        await self.hierarchy.tiers[j].set(key, value)
                return value
        return None

    async def set(self, key: str, value: Any, tier_index: int = 0) -> None:
        """Write value to tier_index, raising ScopeViolationError if not permitted.

        Args:
            key: Cache key.
            value: Value to store.
            tier_index: Target tier index.

        Raises:
            ScopeViolationError: If the agent cannot write to this tier.
        """
        if not self.capability.can_write(tier_index):
            raise ScopeViolationError(
                f"Agent {self.agent_id!r} cannot write to tier {tier_index} "
                f"(permitted tiers: {sorted(self.capability.tiers)})"
            )
        await self.hierarchy.set(key, value, tier_index=tier_index)

    async def delete(self, key: str) -> None:
        """Delete key from all writable tiers this agent can access."""
        for i in range(len(self.hierarchy.tiers)):
            if self.capability.can_write(i):
                await self.hierarchy.tiers[i].delete(key)

    async def exists(self, key: str) -> bool:
        """Return True if key exists in any readable tier."""
        for i in self._readable_tiers():
            if await self.hierarchy.tiers[i].exists(key):
                return True
        return False

"""MemoryAgent — an agent with access to the memory hierarchy."""

from sagemem.hierarchy import MemoryHierarchy


class MemoryAgent:
    """An agent that reads and writes semantic memory through the hierarchy.

    Args:
        agent_id: Unique identifier for this agent.
        hierarchy: The memory hierarchy this agent operates on.
    """

    def __init__(self, agent_id: str, hierarchy: MemoryHierarchy) -> None:
        """Initialize the agent with an ID and a memory hierarchy."""
        self.agent_id = agent_id
        self.hierarchy = hierarchy

    async def remember(self, key: str, value: object, tier_index: int = 0) -> None:
        """Store a value in the hierarchy at the specified tier."""
        raise NotImplementedError("Implemented in Stage 3.")

    async def recall(self, key: str) -> object | None:
        """Retrieve a value from the hierarchy, falling through tiers on miss."""
        raise NotImplementedError("Implemented in Stage 3.")

"""Stage 0 smoke test — verifies all stubs import cleanly and core types exist."""

from sagemem.tiers.base import Tier
from sagemem.coherence.protocol import MESIState, CacheEntry
from sagemem.coherence.bus import CoherenceBus, InvalidateMessage
from sagemem.coherence.merge import last_write_wins, merge_additive
from sagemem.llm.base import LLMClient
from sagemem.hierarchy import MemoryHierarchy
from sagemem.agent import MemoryAgent


def test_mesi_states_exist():
    """All four MESI states must be defined."""
    assert MESIState.Modified
    assert MESIState.Exclusive
    assert MESIState.Shared
    assert MESIState.Invalid


def test_cache_entry_repr():
    """CacheEntry must be constructable and have a readable repr."""
    entry = CacheEntry(key="fact:1", value="the sky is blue", state=MESIState.Shared)
    assert "Shared" in repr(entry)
    assert entry.version == 0


def test_invalidate_message_fields():
    """InvalidateMessage must carry key, writer_id, and new_version."""
    msg = InvalidateMessage(key="fact:1", writer_id="agent-a", new_version=2)
    assert msg.key == "fact:1"
    assert msg.writer_id == "agent-a"
    assert msg.new_version == 2


def test_merge_policies():
    """Merge policies must return expected values."""
    assert last_write_wins("old", "new") == "new"
    assert merge_additive({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_additive({"a": 1}, {"a": 99}) == {"a": 99}  # remote wins on conflict
    assert merge_additive("scalar", "new") == "new"  # fallback to last_write_wins


def test_llm_client_is_protocol():
    """LLMClient must be a runtime-checkable Protocol."""
    # Can't instantiate a Protocol directly — just verify it's importable and checkable
    assert hasattr(LLMClient, "__protocol_attrs__") or hasattr(LLMClient, "_is_protocol")


def test_memory_hierarchy_instantiates():
    """MemoryHierarchy must accept an empty tier list without error."""
    h = MemoryHierarchy(tiers=[])
    assert h.tiers == []


def test_memory_agent_instantiates():
    """MemoryAgent must accept an agent_id, hierarchy, and llm without error."""
    from unittest.mock import AsyncMock
    h = MemoryHierarchy(tiers=[])
    fake_llm = AsyncMock()
    agent = MemoryAgent(agent_id="agent-0", hierarchy=h, llm=fake_llm)
    assert agent.agent_id == "agent-0"

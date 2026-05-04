"""Unit tests for the SageMem LoCoMo adapter.

Verifies that the adapter imports correctly, sets _available=True, and
that _get_hierarchy returns a properly constructed MemoryHierarchy.
Uses a mock DRAMTier to avoid requiring a live Postgres connection.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make locomo_eval src importable
_LOCOMO_SRC = Path(__file__).parents[2] / "locomo_eval" / "src"
if str(_LOCOMO_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCOMO_SRC))

from sagemem.hierarchy import MemoryHierarchy
from sagemem.tiers.l1 import L1Tier


class MockDRAMTier:
    """Minimal stand-in for DRAMTier that skips Postgres."""

    def __init__(self, **kwargs):
        self.embedder = kwargs.get("embedder")
        self._connected = False

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def get(self, key): return None
    async def set(self, key, value): pass
    async def delete(self, key): pass
    async def clear(self): pass
    async def exists(self, key): return False
    async def search(self, embedding, top_k=5): return []


def test_adapter_available_with_sagemem_installed():
    """SageMemAdapter.__init__ should set _available=True when sagemem is importable."""
    with patch("adapters.my_system._make_embedder", return_value=AsyncMock()):
        from adapters.my_system import SageMemAdapter
        adapter = SageMemAdapter()

    assert adapter._available is True


def test_get_hierarchy_returns_memory_hierarchy():
    """_get_hierarchy should build a MemoryHierarchy with L1 + DRAM tiers."""
    import asyncio

    with patch("adapters.my_system._make_embedder", return_value=AsyncMock()):
        # Reload to pick up patched embedder
        import importlib
        import adapters.my_system as mod
        importlib.reload(mod)
        adapter = mod.SageMemAdapter()

    # Patch DRAMTier so connect() doesn't hit Postgres
    adapter._DRAMTier = MockDRAMTier

    h = adapter._get_hierarchy("conv-test-001")

    assert isinstance(h, MemoryHierarchy)
    assert len(h.tiers) == 2
    assert isinstance(h.tiers[0], L1Tier)
    assert isinstance(h.tiers[1], MockDRAMTier)
    assert h.tiers[1]._connected is True


def test_get_hierarchy_reuses_existing():
    """_get_hierarchy should return the same hierarchy for the same conversation_id."""
    with patch("adapters.my_system._make_embedder", return_value=AsyncMock()):
        import importlib
        import adapters.my_system as mod
        importlib.reload(mod)
        adapter = mod.SageMemAdapter()

    adapter._DRAMTier = MockDRAMTier
    h1 = adapter._get_hierarchy("conv-abc")
    h2 = adapter._get_hierarchy("conv-abc")
    assert h1 is h2


def test_reset_clears_and_removes_hierarchy():
    """reset() should call clear on the hierarchy and remove it from cache."""
    with patch("adapters.my_system._make_embedder", return_value=AsyncMock()):
        import importlib
        import adapters.my_system as mod
        importlib.reload(mod)
        adapter = mod.SageMemAdapter()

    adapter._DRAMTier = MockDRAMTier
    adapter._get_hierarchy("conv-reset")
    assert "conv-reset" in adapter._hierarchies

    adapter.reset("conv-reset")
    assert "conv-reset" not in adapter._hierarchies


def test_add_session_returns_stats_when_unavailable():
    """add_session returns empty AddStats when _available is False."""
    with patch("adapters.my_system._make_embedder", return_value=AsyncMock()):
        import importlib
        import adapters.my_system as mod
        importlib.reload(mod)
        adapter = mod.SageMemAdapter()

    adapter._available = False
    stats = adapter.add_session("c", "s", "2024-01-01", [])
    assert stats.latency_ms == 0.0


def test_search_returns_empty_when_unavailable():
    """search returns empty SearchResult when _available is False."""
    with patch("adapters.my_system._make_embedder", return_value=AsyncMock()):
        import importlib
        import adapters.my_system as mod
        importlib.reload(mod)
        adapter = mod.SageMemAdapter()

    adapter._available = False
    result = adapter.search("c", "query")
    assert result.context == ""
    assert result.retrieved_tokens == 0

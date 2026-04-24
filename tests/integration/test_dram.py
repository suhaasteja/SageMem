"""Integration tests for DRAM (Postgres + pgvector) tier. Requires pgvector extension."""

import pytest

from sagemem.tiers.dram import DRAMTier

PG_DSN = "postgresql://localhost/sagemem_test"
EMBED_DIM = 4  # small dimension for tests


def fake_embedder(dim: int):
    """Returns an async embedder that produces a fixed-length vector."""
    async def embed(text: str) -> list[float]:
        # Deterministic fake: hash text into a unit-ish vector
        base = float(hash(text) % 1000) / 1000.0
        return [base + i * 0.01 for i in range(dim)]
    return embed


@pytest.fixture
async def dram():
    tier = DRAMTier(
        dsn=PG_DSN,
        table="test_dram",
        embedding_dim=EMBED_DIM,
        embedder=fake_embedder(EMBED_DIM),
    )
    await tier.connect()
    await tier.clear()
    yield tier
    await tier.clear()
    await tier.disconnect()


async def test_set_and_get(dram):
    await dram.set("k1", "semantic memory")
    assert await dram.get("k1") == "semantic memory"


async def test_get_missing_returns_none(dram):
    assert await dram.get("ghost") is None


async def test_delete(dram):
    await dram.set("k1", "v")
    await dram.delete("k1")
    assert await dram.get("k1") is None


async def test_exists(dram):
    assert not await dram.exists("k1")
    await dram.set("k1", "v")
    assert await dram.exists("k1")


async def test_clear(dram):
    await dram.set("a", 1)
    await dram.set("b", 2)
    await dram.clear()
    assert await dram.get("a") is None


async def test_semantic_search_returns_results(dram):
    await dram.set("fact:sky", "the sky is blue")
    await dram.set("fact:grass", "grass is green")
    query_vec = [0.1, 0.2, 0.3, 0.4]
    results = await dram.search(query_embedding=query_vec, top_k=2)
    assert len(results) <= 2
    for r in results:
        assert "key" in r
        assert "value" in r
        assert "distance" in r


async def test_stores_dict_value(dram):
    payload = {"content": "agent memory", "source": "research"}
    await dram.set("doc:1", payload)
    assert await dram.get("doc:1") == payload

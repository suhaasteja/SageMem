# SageMem

A GPU-memory-hierarchy-inspired memory layer for multi-agent LLM systems.

Four memory tiers (L1/L2/L3/DRAM), a MESI-style coherence protocol for shared beliefs, and capacity-based permission scoping — applied to semantic memory at the orchestration layer.

![SageMem architecture](https://github.com/user-attachments/assets/f8cba28a-0595-44dd-8396-aa37b1b032a9)

---

## The problem

Multi-agent LLM systems have a memory problem that looks exactly like the one CPUs solved decades ago: many parallel compute units sharing state, with no principled protocol for who sees what updates when.

Existing frameworks (LangGraph, Mem0, Letta) handle concurrent writes with last-write-wins or ad-hoc reducers. Stale reads are silent. There is no cache coherence model in production multi-agent memory anywhere.

## The approach

The GPU memory hierarchy was designed from day one for many parallel processors sharing state with bounded latency and explicit coherence. SageMem applies that model to semantic memory.

| Tier | Analog              | Storage              | Scope        | Latency  |
|------|---------------------|----------------------|--------------|----------|
| L1   | SM registers + L1   | In-memory LRU        | Per-agent    | ~0.000ms |
| L2   | Shared L2 per GPC   | Redis                | Per-cluster  | ~0.2ms   |
| L3   | Global shared memory| Postgres JSONB       | Global       | ~0.5ms   |
| DRAM | HBM / VRAM          | Postgres + pgvector  | Global/semantic | ~0.4ms |

**MESI coherence:** Each L1 cache entry carries a state (Modified / Exclusive / Shared / Invalid). When an agent writes a shared key, it broadcasts an `InvalidateMessage` over Redis pub/sub. All agents with that key cached receive the message and mark their entry `Invalid`. The next read re-fetches the fresh value.

**Capacity as capability:** Each agent declares which tiers it may access. `ScopedHierarchy` enforces this as a hard boundary — an agent restricted to L1+L2 physically cannot see L3 data.

---

## Benchmark results

```
Debate workload (2 agents, conflicting writes):
  Tier-aware : 50 coherence invalidations fired — stale reads detected and corrected
  Flat memory: 0 invalidations — stale reads persist silently

Parallel search (4 agents, overlapping key sets):
  Tier-aware : 43% of reads served from shared L2 (cross-agent cache hits)
  Flat memory: 0% — each agent is an island, work cannot be shared
```

---

## Installation

Requires Python 3.12+, Redis, and Postgres with pgvector.

```bash
# Install dependencies
brew install redis postgresql pgvector
brew services start redis
brew services start postgresql
createdb sagemem_test
psql sagemem_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Install the package
uv sync --extra dev
```

---

## Usage

### Basic hierarchy

```python
import asyncio
from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.hierarchy import MemoryHierarchy

async def main():
    l1 = L1Tier(capacity=128)
    l2 = L2Tier(url="redis://localhost:6379", namespace="myapp")
    l3 = L3Tier(dsn="postgresql://localhost/mydb")

    await l2.connect()
    await l3.connect()

    h = MemoryHierarchy(tiers=[l1, l2, l3])

    # Write to L3; read falls through L1→L2→L3, promotes on hit
    await h.set("fact:pi", 3.14159, tier_index=2)
    value = await h.get("fact:pi")   # L1 miss → L2 miss → L3 hit → promotes to L1+L2
    value = await h.get("fact:pi")   # L1 hit

asyncio.run(main())
```

### Coherence protocol

```python
from sagemem.coherence.bus import CoherenceBus
from sagemem.tiers.l1_coherent import CoherentL1Tier

bus = CoherenceBus(url="redis://localhost:6379")
await bus.connect()

l1_a = CoherentL1Tier(agent_id="agent-a", bus=bus)
l1_b = CoherentL1Tier(agent_id="agent-b", bus=bus)
await l1_a.start()  # subscribes to invalidation messages
await l1_b.start()

# Agent B caches a value
await l1_b.set("belief:x", "old value")

# Agent A writes — broadcasts InvalidateMessage
await l1_a.set("belief:x", "new value")
# l1_b's entry for "belief:x" is now Invalid
# Next read by agent B re-fetches from L2/L3
```

### Capacity scoping

```python
from sagemem.scope import AgentCapability, ScopedHierarchy, ScopeViolationError

cap = AgentCapability(
    tiers={0, 1},       # L1 + L2 only
    l3_read=False,
    l3_write=False,
)
scoped = ScopedHierarchy(hierarchy, cap, agent_id="restricted-agent")

await scoped.get("l3_key")    # returns None — L3 not in scope
await scoped.set("k", "v", tier_index=2)  # raises ScopeViolationError
```

---

## Running tests

```bash
# Unit tests (no services needed)
uv run pytest tests/unit/ -v

# Integration tests (requires Redis + Postgres)
uv run pytest tests/integration/ -v

# Hypothesis property tests (MESI coherence invariants)
uv run pytest tests/property/ -v

# All tests
uv run pytest tests/ -q
```

---

## Benchmarks

```bash
# Tier latency
uv run python benchmarks/tier_latency.py

# Full workload comparison vs flat-memory baseline
uv run python -m benchmarks.run_all
```

---

## Live demo

```bash
uv run python -m demo.server.main
# Open http://localhost:8000
```

The demo shows two agents writing conflicting beliefs, the MESI invalidation firing over Redis pub/sub, and both agents converging to the same value in real time.

---

## Documentation

- [Architecture](docs/architecture.md) — tier design, read/write paths, design decisions
- [Coherence protocol](docs/coherence-protocol.md) — MESI state machine, Redis transport, L3 CAS
- [Benchmarks](docs/benchmarks.md) — methodology, results, interpretation

---

## Academic context

This project implements and extends the framework proposed in:

> Yu et al., *"Multi-Agent Memory from a Computer Architecture Perspective"*, arxiv:2603.10062, March 2026

The paper proposes a three-tier hierarchy and identifies two protocol gaps: cache sharing across agents, and structured memory access control. SageMem is an implementation of that framework with two concrete additions:

1. A MESI-inspired coherence protocol with Redis pub/sub invalidation
2. Capacity as a capability/permission boundary enforced at the API level

---

## What this is not

- Not a production-grade distributed system (no multi-region, no HA)
- Not an inference engine modification (we do not touch KV cache tensors)
- Not LangGraph, Mem0, or Letta — those are what this is designed to replace

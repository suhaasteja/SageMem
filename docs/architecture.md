# SageMem Architecture

## The problem

Multi-agent LLM systems have a memory problem that looks exactly like the one CPUs and GPUs solved decades ago:

- Many parallel compute units (agents) need to share state
- Fast local memory (context window) is small; slow global memory (vector stores) is large
- Concurrent writes to shared state create coherence bugs
- No existing framework has a principled protocol for who sees what updates when

Existing multi-agent frameworks (LangGraph, Mem0, Letta) have at most two tiers (working + persistent) and handle concurrent writes with "last write wins" or ad-hoc reducers. There is no real coherence model in production anywhere.

## The insight

The GPU memory hierarchy is the only established model designed from day one for many parallel processors sharing state with bounded latency and explicit coherence. SageMem applies that model to semantic memory at the orchestration layer.

## Layer diagram

```
┌──────────────────────────────────────────────────┐
│  Application / Agent framework                    │
├──────────────────────────────────────────────────┤
│  SageMem                                          │
│  ├── ScopedHierarchy  (capability enforcement)    │
│  ├── MemoryHierarchy  (tiered read/write)         │
│  ├── CoherenceBus     (Redis pub/sub invalidation)│
│  └── Tiers: L1 / L2 / L3 / DRAM                 │
├──────────────────────────────────────────────────┤
│  Infrastructure                                   │
│  ├── Redis   (L2 cache + pub/sub)                 │
│  └── Postgres + pgvector  (L3 + DRAM)            │
├──────────────────────────────────────────────────┤
│  LLM API  (Anthropic / OpenAI / local)           │
└──────────────────────────────────────────────────┘
```

SageMem sits entirely at the orchestration layer. It does not touch inference engines, KV cache tensors, or model weights.

## The four tiers

| Tier | GPU analog           | Storage                  | Scope        | Latency   |
|------|----------------------|--------------------------|--------------|-----------|
| L1   | SM registers + L1    | In-memory LRU (`L1Tier`) | Per-agent    | ~0.001ms  |
| L2   | Shared L2 per GPC    | Redis (`L2Tier`)         | Per-cluster  | ~0.2ms    |
| L3   | Global shared memory | Postgres JSONB (`L3Tier`)| Global       | ~0.5ms    |
| DRAM | HBM / VRAM           | Postgres + pgvector      | Global/semantic | ~0.4ms |

Measured latencies (200-iteration benchmark, local machine):

```
L1  read  median=0.000ms  p95=0.000ms
L2  read  median=0.155ms  p95=0.227ms
L3  read  median=0.303ms  p95=0.487ms
DRAM read median=0.270ms  p95=0.431ms
```

## Read path

A `MemoryHierarchy.get(key)` call falls through tiers in order:

```
L1 → hit? return value + promote to nothing (already fastest)
   ↓ miss
L2 → hit? return value + promote to L1
   ↓ miss
L3 → hit? return value + promote to L1, L2
   ↓ miss
DRAM → hit? return value + promote to L1, L2, L3
     ↓ miss → return None
```

Promotion is always to all faster tiers simultaneously, not lazy. The next read on the same key always hits L1.

## Write path

Writes go to a specified tier only. The caller (agent or higher-level policy) decides what tier a fact belongs in. This is deliberate — automatic write propagation across tiers would bypass the capability model.

## Capacity as capability

Each agent declares an `AgentCapability`:

```python
AgentCapability(
    tiers={0, 1},          # L1 + L2 only
    l2_read=True,
    l2_write=True,
    l3_read=False,         # cannot see L3
    l3_write=False,        # cannot write to L3
)
```

`ScopedHierarchy` wraps `MemoryHierarchy` and enforces this. An agent restricted to L1+L2 cannot physically see L3 data — its `get()` never queries L3. Attempts to write to out-of-scope tiers raise `ScopeViolationError` immediately.

This makes tier design a safety feature, not just a performance feature.

## Design decisions

**Hand-rolled orchestration, not LangGraph.** LangGraph's reducers implement last-write-wins with no coherence model. Using it would force us to fight the abstraction we're replacing.

**Redis for L2.** Atomic pub/sub for invalidation messages. `WATCH`/`MULTI` for compare-and-swap. Fast enough that the L1→L2 latency gap is real.

**Postgres for L3 and DRAM.** Real transactions for L3 version conflicts. pgvector gives semantic search without a separate vector DB.

**Python 3.12 + asyncio.** Coherence bugs only surface under concurrent IO. Sync code cannot stress the protocol meaningfully.

## What we are not

- Not a fork of vLLM, TGI, or any inference engine
- Not managing GPU KV cache tensors (that is a v2 concern)
- Not LangGraph, Mem0, or Letta — those are what this replaces

# SageMem Coherence Protocol

## The problem it solves

When two agents hold a cached copy of the same belief and one agent updates it, the other agent's copy becomes stale. Without a protocol, stale reads persist silently — the agent continues reasoning on outdated information with no indication that anything is wrong.

The debate workload benchmark demonstrates this concretely: with a flat-memory baseline, 50 potential staleness events go undetected. With SageMem's coherence protocol, each of those events fires an invalidation and the stale reader is forced to re-fetch.

## MESI state machine

Each entry in L1 carries one of four states:

| State     | Meaning |
|-----------|---------|
| Modified  | Local has a write not yet seen by other agents |
| Exclusive | Local matches shared tier; no other agent has it cached |
| Shared    | Local matches shared tier; other agents may also have it cached |
| Invalid   | Entry is stale; must re-fetch from the next tier before use |

### State transitions

```
                  local_write
    ┌─────────────────────────────────────┐
    │                                     ▼
    │   local_read          ┌─────────── Modified ◄──── local_write ────┐
    │  ┌────────┐           │            │                               │
    │  │        │     remote_write       │ remote_write / evict          │
    │  ▼        │           │            ▼                               │
  Exclusive ────┘           │          Invalid ──── fetch ──────► Shared ┘
    │                       │            │                         │
    │ local_write           └──────────► │                         │ local_read
    │                                    │                         └────────┐
    ▼                                    │                                  │
  Modified ◄──── local_write ─── Shared ◄┘                                 ▼
                                  │                                       Shared
                                  │ remote_write / evict
                                  ▼
                                Invalid
```

Simplified transitions:

- **Any state + `remote_write`** → Invalid (another agent wrote; we are stale)
- **Any state + `local_write`** → Modified (we have unseen changes)
- **Exclusive + `local_read`** → Exclusive (still clean, still exclusive)
- **Shared + `local_read`** → Shared (still clean, shared)
- **Invalid + `fetch`** → Shared (re-fetched from shared tier)
- **Any + `evict`** → Invalid (entry removed from local cache)

All 14 transitions are tested exhaustively in `tests/unit/test_mesi.py`. Hypothesis runs 500 randomized event sequences in `tests/property/test_coherence.py` and finds no counterexamples.

## Transport: Redis pub/sub

The `CoherenceBus` uses Redis pub/sub on a shared channel (`sagemem:coherence`).

### Write flow

```
Agent A writes key K:
  1. A updates local L1 entry → state = Modified
  2. A publishes InvalidateMessage(key=K, writer_id="A", new_version=N)
     to the coherence channel
  3. All subscribed agents receive the message
  4. Each agent with K in its L1 cache:
       - Ignores if writer_id == self.agent_id (self-filter)
       - Otherwise: entry.state = Invalid
  5. Next time another agent reads K:
       - L1 returns None (Invalid treated as miss)
       - Read falls through to L2/L3
       - Fresh value is returned and promoted back to L1
```

### Message format

```json
{
  "key": "belief:climate",
  "writer_id": "agent-a",
  "new_version": 2
}
```

### Why always broadcast?

The writer does not need to know who else has K cached. Broadcast invalidation is simpler than directory-based tracking and correct: agents that do not have K cached simply ignore the message.

## L3 optimistic concurrency

L3 (Postgres) uses a `version` integer column for compare-and-swap (CAS) writes:

```sql
UPDATE sagemem_l3
   SET value = $1, version = version + 1
 WHERE key = $2 AND version = $3   -- only update if version matches
```

If the update touches 0 rows, the caller's expected version is stale — another writer got there first. SageMem raises `VersionConflictError`. The caller retries with a fresh read.

This prevents the "lost update" problem: two agents reading version 1, both computing an update, and one silently overwriting the other. With CAS, only one wins; the other gets an explicit error and must reconcile.

## What the protocol does NOT do

- **No directory tracking.** We do not track which agents have which keys cached. Broadcast invalidation handles this at the cost of some unnecessary messages.
- **No L2 coherence.** L2 (Redis) is a shared tier, not per-agent. Stale L2 entries are overwritten by the next write to that key. Full L2 MESI would require tracking per-agent Redis namespaces separately — a v2 concern.
- **No causal ordering.** Messages may arrive out of order under high load. The protocol is correct at the entry level (each entry converges) but does not guarantee global causal consistency.

## Comparison to existing approaches

| Approach | Staleness detection | Concurrent write handling |
|----------|--------------------|-----------------------------|
| LangGraph reducers | None (last-write-wins) | Reducer merge (user-defined) |
| Mem0 | None | Last-write-wins |
| SageMem | Explicit invalidation (MESI) | L3 optimistic CAS + retry |
| Hardware MESI | Explicit invalidation | Bus snooping / directory |

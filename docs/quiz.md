# SageMem Quiz

Mix of multiple choice, true/false, short answer, and code reading. No peeking at the source.

---

## Part 1 — Motivation & Origin (5 questions)

**Q1.** What was the original tweet insight that led to SageMem, and what academic paper independently validated the same direction? Name the paper, authors, and arXiv ID.

**Q2.** The project is named "SageMem." In GPU architecture, what is a *warp* (the term SageMem draws from)? What property of warps makes it the right analogy for an agent cluster?

**Q3.** True or False: SageMem modifies the inference engine (e.g., vLLM) to share KV cache tensors between agents at the GPU level.

**Q4.** Name three multi-agent frameworks that SageMem is explicitly *not* built on top of, and briefly explain the specific weakness of each that SageMem is designed to fix.

**Q5.** The UCSD paper identified two concrete protocol gaps in existing multi-agent memory systems. What were they, and how does SageMem address each one?

---

## Part 2 — The Four Tiers (8 questions)

**Q6.** Match each tier to its GPU memory analog:

| SageMem tier | GPU analog |
|---|---|
| L1 | ? |
| L2 | ? |
| L3 | ? |
| DRAM | ? |

Options: *HBM/VRAM, SM registers + L1, Global shared memory, Shared L2 per GPC*

**Q7.** What storage backend does each tier use? Fill in:
- L1: ___
- L2: ___
- L3: ___
- DRAM: ___

**Q8.** From the benchmark results, what is the approximate read latency ratio between L1 and L2? Between L1 and L3?

**Q9.** True or False: DRAM tier latency is consistently higher than L3 in the benchmark results, as you'd expect from the GPU analogy.

**Q10.** What Python data structure underlies L1? What eviction policy does it use, and how is that policy implemented in the code?

**Q11.** L3 stores values as `JSONB` in Postgres. It also has an extra column that L2 does not have. What is that column, what type is it, and what is it used for?

**Q12.** What Python package is used for async Postgres access? What package for async Redis?

**Q13.** DRAM tier supports a capability that L1/L2/L3 do not. What is it, and what infrastructure enables it?

---

## Part 3 — The Read Path (6 questions)

**Q14.** Describe the full read path when `MemoryHierarchy.get("key")` is called and the key exists only in L3. What happens step by step, including what changes in which tiers after the read completes?

**Q15.** After a value is promoted from L3 to L1, you delete it from L3. What happens on the next `MemoryHierarchy.get()` call for that key? Walk through each tier.

**Q16.** True or False: When a value is found at L2, it gets promoted to L1 AND written back to L3 to keep L3 fresh.

**Q17.** What metric does `MemoryHierarchy` emit to Prometheus when a key is found at L2 after missing L1? Name the specific counter and its labels.

**Q18.** The `MemoryHierarchy.set()` method takes a `tier_index` parameter. What happens if you call `h.set("k", "v")` without specifying `tier_index`? What tier does it write to?

**Q19.** True or False: `MemoryHierarchy.get()` will automatically write-back a promoted value to tiers *slower* than the hit tier, to keep them in sync.

---

## Part 4 — MESI Coherence Protocol (10 questions)

**Q20.** What does MESI stand for? Write out the full name for each letter and one sentence describing what each state means in the context of SageMem.

**Q21.** Fill in the state transitions table:

| From state | Event | To state |
|---|---|---|
| Exclusive | local_write | ? |
| Shared | remote_write | ? |
| Modified | remote_write | ? |
| Invalid | fetch | ? |
| Shared | local_read | ? |
| Invalid | local_write | ? |
| Exclusive | evict | ? |
| Modified | local_read | ? |

**Q22.** Which event is *not* valid from the `Modified` state — `fetch` or `local_read`? Why?

**Q23.** A `CoherentL1Tier` entry is in `Shared` state. Agent A writes a new value to that key. Walk through exactly what happens: which events fire, which states change, and in which order.

**Q24.** True or False: When agent A publishes an `InvalidateMessage` with `writer_id="agent-a"`, agent A's own `CoherentL1Tier` callback processes the message and transitions its own entry to `Invalid`.

**Q25.** In the original implementation (before the bug fix in Stage 4), `CoherentL1Tier.set()` only published an invalidation under what condition? Why was this wrong, and what was the fix?

**Q26.** What is the structure of an `InvalidateMessage`? Name its three fields and their types.

**Q27.** The Hypothesis property tests in `tests/property/test_coherence.py` test five invariants. Name at least three of them.

**Q28.** What Redis feature does `CoherenceBus` use for transport? How does it handle multiple agents subscribing — is there one connection or two, and why?

**Q29.** True or False: The MESI protocol in SageMem guarantees global causal ordering of writes — if A writes before B, all agents will see A's value before B's.

---

## Part 5 — L3 Optimistic Concurrency (4 questions)

**Q30.** Explain the compare-and-swap (CAS) write pattern used in L3. What SQL does `set_versioned()` execute, and how does it detect a conflict?

**Q31.** Two agents both read key `"belief:x"` at version 3. Agent A calls `set_versioned("belief:x", "A's value", expected_version=3)` and succeeds. Then agent B calls `set_versioned("belief:x", "B's value", expected_version=3)`. What happens and why?

**Q32.** What exception does SageMem raise on a version conflict, and where in the codebase is it defined?

**Q33.** The regular `L3Tier.set()` (not `set_versioned`) also increments the version. What SQL pattern does it use, and when would you use it over `set_versioned`?

---

## Part 6 — Capacity as Capability (6 questions)

**Q34.** What class enforces tier access boundaries, and what exception does it raise on a violation?

**Q35.** An agent is instantiated with this capability:
```python
AgentCapability(
    tiers={0, 1},
    l2_read=True,
    l2_write=False,
    l3_read=False,
    l3_write=False,
)
```
For each of the following calls on a `ScopedHierarchy`, state whether it succeeds, returns None, or raises — and why:
- `await scoped.get("key")` when key exists only in L3
- `await scoped.get("key")` when key exists in L2
- `await scoped.set("key", "val", tier_index=1)`
- `await scoped.set("key", "val", tier_index=2)`
- `await scoped.delete("key")`

**Q36.** When `ScopedHierarchy.get()` finds a value at tier 2 (L3) and promotes it, which tiers does it promote to? Does it promote to all faster tiers, or only the ones the agent is allowed to write to?

**Q37.** True or False: `AgentCapability` validates at construction time that `tiers` matches the read/write flags. For example, if you set `tiers={0}` but `l2_read=True`, it raises an error.

**Q38.** How does the `parallel_search.py` example demonstrate hard isolation vs advisory isolation? Describe what beta agent can and cannot do.

**Q39.** In `AgentCapability`, tier index 0 is always readable and writable if it's in `tiers`. Why is L1 treated differently from L2/L3/DRAM (which have separate read/write flags)?

---

## Part 7 — Code Reading (5 questions)

**Q40.** Read this snippet:
```python
async def get(self, key: str) -> Any | None:
    if key not in self._store:
        return None
    entry = self._store[key]
    if entry.state == MESIState.Invalid:
        return None
    self._store.move_to_end(key)
    apply_event(entry, "local_read")
    return entry.value
```
What class is this from? What is `self._store`? Why does it return `None` for `Invalid` entries instead of returning `entry.value`?

**Q41.** Read this snippet:
```python
result = await self._pool_or_raise().execute(f"""
    UPDATE {self.table}
       SET value = $1::jsonb,
           version = version + 1
     WHERE key = $2 AND version = $3
""", serialized, key, expected_version)
if result == "UPDATE 0":
    try:
        await self._pool_or_raise().execute(f"""
            INSERT INTO {self.table} (key, value, version)
            VALUES ($1, $2::jsonb, 0)
        """, key, serialized)
    except asyncpg.UniqueViolationError:
        raise VersionConflictError(...)
```
Why does the code attempt an `INSERT` when the `UPDATE` returns `"UPDATE 0"`? When does the `INSERT` succeed vs raise `UniqueViolationError`?

**Q42.** The `MemoryHierarchy.get()` method uses `time.perf_counter()` around each tier read. What does it do with that timing, and what Prometheus instrument does it use?

**Q43.** What does `bus.publish_invalidate = counting_publish` do in the debate benchmark? Is this a valid Python pattern — why or why not?

**Q44.** Look at the `CoherentL1Tier.set_shared()` method signature:
```python
async def set_shared(self, key: str, value: Any, version: int = 0) -> None:
```
How does it differ from `set()`? When would the hierarchy call `set_shared()` instead of `set()`?

---

## Part 8 — Benchmarks & Design (6 questions)

**Q45.** The debate workload shows 50 coherence invalidations. Given 2 agents, 5 belief keys, and 5 rounds where each agent writes all keys — derive the expected number of invalidations mathematically.

**Q46.** The parallel search workload reports 43% of reads served from L2 cross-agent cache. Given 4 agents each searching 8 keys from a 16-key corpus with a specific overlap pattern — explain why not all reads are L2 hits (why isn't it 100%)?

**Q47.** The research workload shows identical token counts (66) for tier-aware and flat-memory. The PLAN.md says token savings is a core benefit. Resolve this apparent contradiction — when and why would the numbers diverge?

**Q48.** The benchmark uses a `MockLLM` instead of real Anthropic API calls. Give two reasons why this is the correct choice for a memory-layer benchmark.

**Q49.** The debate workload takes ~280ms while the flat baseline takes ~0ms. Is this a fair comparison? What is the 280ms actually measuring, and what does it mean for real-world use?

**Q50.** Three teams review SageMem and each raises a concern:
- **Team A:** "You should use LangGraph's checkpointing for L3 instead of raw Postgres."
- **Team B:** "Your L2 coherence is incomplete — L2 (Redis) entries can be stale after an invalidation."
- **Team C:** "The MESI protocol only covers L1. What stops two agents from concurrently writing to L3 via the hierarchy directly?"

For each concern, state whether it is valid, partially valid, or not valid — and explain.

---

## Scoring guide

| Score | Assessment |
|---|---|
| 45–50 | You could give a conference talk on this |
| 35–44 | Solid understanding, a few gaps to close |
| 25–34 | Good grasp of concepts, weaker on implementation details |
| 15–24 | Understand the motivation, need to re-read the code |
| < 15 | Start with `docs/architecture.md` and re-read the CLAUDE.md |

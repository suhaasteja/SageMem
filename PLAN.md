# SageMem — Development Plan

This is the working reference guide for building SageMem stage by stage.

**Rules:**
1. One stage at a time. Do not start the next stage until the current one is confirmed working.
2. Each stage ends with a "How to verify" section — the user runs these and confirms.
3. Once confirmed, push to GitHub before moving on.
4. If a stage breaks a previous stage's tests, fix it before proceeding.

---

## Stage 0 — Skeleton + Typed Interfaces

**Goal:** Repo is set up, all abstractions exist as stubs, one smoke test passes.

**Deliverables:**
- `pyproject.toml` with all dependencies declared (even if not yet used)
- `src/sagemem/tiers/base.py` — abstract `Tier` class with `get`, `set`, `delete`, `clear`
- `src/sagemem/coherence/bus.py` — stubbed `CoherenceBus`
- `src/sagemem/coherence/protocol.py` — `MESIState` enum (Modified, Exclusive, Shared, Invalid)
- `src/sagemem/agent.py` — stubbed `MemoryAgent`
- `src/sagemem/hierarchy.py` — stubbed `MemoryHierarchy`
- `src/sagemem/llm/base.py` — `LLMClient` protocol (abstract interface)
- `tests/unit/test_smoke.py` — one test that imports everything and asserts nothing crashes

**How to verify:**
```bash
uv run pytest tests/unit/test_smoke.py -v
```
Expected: 1 test passes, no import errors.

**Status:** [ ] In progress / [x] Done / [ ] Confirmed by user

---

## Stage 1 — Four Tiers Working in Isolation

**Goal:** Each tier can store and retrieve data independently. Latency is measurable.

**Deliverables:**
- `src/sagemem/tiers/l1.py` — in-memory LRU with configurable capacity
- `src/sagemem/tiers/l2.py` — Redis-backed tier (async)
- `src/sagemem/tiers/l3.py` — Postgres JSONB + `version` column (async)
- `src/sagemem/tiers/dram.py` — Postgres + pgvector (async)
- `tests/unit/test_l1.py` — LRU eviction, get/set/delete
- `tests/integration/test_l2.py` — requires live Redis
- `tests/integration/test_l3.py` — requires live Postgres
- `tests/integration/test_dram.py` — requires pgvector extension
- `benchmarks/tier_latency.py` — prints median latency for each tier

**How to verify:**
```bash
# Unit tests (no external services needed)
uv run pytest tests/unit/ -v

# Integration tests (requires Redis + Postgres running)
uv run pytest tests/integration/ -v

# Latency benchmark — should show L1 < L2 < L3 ≈ DRAM
uv run python benchmarks/tier_latency.py
```
Expected: All tests pass. Benchmark output shows distinct latency tiers.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 2 — Hierarchical Read Path

**Goal:** A single read request falls through tiers (L1 → L2 → L3 → DRAM) and promotes on miss.

**Deliverables:**
- `src/sagemem/hierarchy.py` — `MemoryHierarchy` with `get` (fall-through + promotion) and `set` (write to target tier)
- `src/sagemem/metrics.py` — Prometheus counters: `tier_hits_total`, `tier_misses_total` per tier
- `tests/unit/test_hierarchy.py` — mock tiers, verify fall-through order and promotion behavior
- `tests/integration/test_hierarchy_e2e.py` — live tiers, write to L3, read via hierarchy, confirm L1 is populated

**How to verify:**
```bash
uv run pytest tests/unit/test_hierarchy.py -v
uv run pytest tests/integration/test_hierarchy_e2e.py -v
```
Then run the e2e test with a debug print showing which tier served each read. Expected: first read hits L3, second read hits L1.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 3 — Single-Agent Demo

**Goal:** One agent uses the hierarchy for context assembly. Measurably better than flat memory.

**Deliverables:**
- `src/sagemem/llm/anthropic.py` — `AnthropicClient` implementing `LLMClient`
- `src/sagemem/agent.py` — `MemoryAgent` with `remember(key, value, tier)` and `recall(key)` using the hierarchy
- `examples/research_assistant.py` — single agent answers questions, stores facts in hierarchy, reuses them
- Baseline comparison: same task with a flat dict (no tiers), print token usage side by side

**How to verify:**
```bash
# Requires ANTHROPIC_API_KEY in environment
uv run python examples/research_assistant.py
```
Expected: Agent answers a multi-turn question. Second turn reuses cached facts from L1 (visible in logs). Token usage printed at end — tier-aware should use fewer tokens on recall.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 4 — Coherence Protocol (Core Contribution)

**Goal:** Two agents writing concurrently to a shared key trigger MESI invalidation and converge correctly.

**Deliverables:**
- `src/sagemem/coherence/protocol.py` — full MESI state machine with transitions
- `src/sagemem/coherence/bus.py` — Redis pub/sub transport: `publish_invalidate`, `subscribe`, listener loop
- `src/sagemem/coherence/merge.py` — conflict resolution policies: `last_write_wins`, `merge_additive`
- L3 optimistic concurrency: `version` column CAS on every write, retry on conflict
- `tests/unit/test_mesi.py` — state machine transitions, all valid state changes
- `tests/property/test_coherence.py` — Hypothesis: two agents writing concurrently always converge to same value
- `tests/integration/test_coherence_e2e.py` — live Redis + Postgres, two agents, confirm no stale reads after convergence

**How to verify:**
```bash
uv run pytest tests/unit/test_mesi.py -v
uv run pytest tests/property/test_coherence.py -v  # runs 100 Hypothesis examples
uv run pytest tests/integration/test_coherence_e2e.py -v
```
Expected: All pass. Hypothesis finds no counterexamples. Integration test logs show Invalidate events and state transitions.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 5 — Capacity as Capability

**Goal:** Agents are scoped to declared tiers. A scoped agent physically cannot see data outside its capability.

**Deliverables:**
- `src/sagemem/scope.py` — `AgentCapability` dataclass, `ScopedHierarchy` wrapper that raises on out-of-scope access
- `tests/unit/test_scope.py` — agent with L1+L2 capability cannot read from L3/DRAM; raises `ScopeViolationError`
- `examples/parallel_search.py` — two agents with different scopes operating on same hierarchy, demonstrating isolation

**How to verify:**
```bash
uv run pytest tests/unit/test_scope.py -v
uv run python examples/parallel_search.py
```
Expected: Scope tests confirm violations raise. Example shows two agents, one restricted, operating independently with clear log output of what each can and cannot access.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 6 — Benchmark Harness

**Goal:** Numbers exist. Tier-aware beats flat-memory on token usage and staleness.

**Deliverables:**
- `benchmarks/workloads/research.py` — multi-agent research task
- `benchmarks/workloads/debate.py` — two agents debating, sharing beliefs
- `benchmarks/workloads/parallel_search.py` — parallel agents merging findings
- `benchmarks/baselines/flat_memory.py` — same tasks with a plain dict
- Results printed as a table: hit rate, token usage, staleness, L3 conflict retries

**How to verify:**
```bash
uv run python benchmarks/run_all.py
```
Expected: Table output comparing sagemem vs flat baseline. Sagemem shows higher hit rates and lower token usage on repeat queries. Numbers are real (not mocked).

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 7 — Live Visualizer + Demo Recording

**Goal:** 60-second demo that a non-technical person can follow.

**Deliverables:**
- `demo/server/` — FastAPI backend serving coherence events as SSE stream
- `demo/ui/` — single-page app: three agent panels, shared tier panels in center, animated invalidation events
- Demo script: two agents write conflicting values, coherence resolves, visualizer shows it live

**How to verify:**
```bash
uv run python demo/server/main.py &
open http://localhost:8000
```
Expected: UI loads, agents animate, coherence events visible in real time. Record a 60-second screen capture.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Stage 8 — Writeup + Clean Public Repo

**Goal:** Repo is presentable. Blog post and short paper draft exist.

**Deliverables:**
- `docs/architecture.md` — tier diagram, design decisions
- `docs/coherence-protocol.md` — MESI adaptation, state diagram, Redis transport
- `docs/benchmarks.md` — results from Stage 6 with interpretation
- `README.md` updated — what it is, why it matters, how to run, link to writeup
- Blog post draft (external, not in repo)

**How to verify:**
- Fresh clone, follow README from scratch, all stages run without extra explanation needed.

**Status:** [ ] In progress / [ ] Done / [ ] Confirmed by user

---

## Current Stage

**Stage 8** — complete. All stages shipped.

---

## GitHub Push Log

| Stage | Description                                  | Pushed |
|-------|----------------------------------------------|--------|
| 0     | Skeleton + typed interfaces, smoke tests     | ✓      |
| 1     | Four tiers in isolation + latency benchmark  | ✓      |
| 2     | Hierarchical read path + Prometheus metrics  | ✓      |
| 3     | Single-agent demo + Anthropic LLM client     | ✓      |
| 4     | MESI coherence protocol + Hypothesis tests   | ✓      |
| 5     | Capacity as capability + ScopedHierarchy     | ✓      |
| 6     | Benchmark harness (3 workloads vs baseline)  | ✓      |
| 7     | Live visualizer (FastAPI + SSE + HTML/JS)    | ✓      |
| 8     | Docs + clean README                          | ✓      |

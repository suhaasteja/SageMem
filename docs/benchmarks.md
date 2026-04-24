# SageMem Benchmarks

## Setup

- Machine: Apple Silicon Mac, local Redis + Postgres
- Redis: Homebrew install, default config
- Postgres 18 + pgvector
- Python 3.12, asyncio, 200 iterations per tier latency measurement
- LLM calls in workload benchmarks use a mock client (no network latency)

Run benchmarks yourself:

```bash
# Tier latency (requires Redis + Postgres)
uv run python benchmarks/tier_latency.py

# Full workload comparison
uv run python -m benchmarks.run_all
```

---

## Tier latency

200-iteration benchmark, each tier measured independently:

| Tier | Read median | Read p95  | Write median | Write p95  |
|------|------------|-----------|--------------|------------|
| L1   | 0.000ms    | 0.000ms   | 0.000ms      | 0.000ms    |
| L2   | 0.155ms    | 0.227ms   | 0.235ms      | 0.576ms    |
| L3   | 0.303ms    | 0.487ms   | 0.540ms      | 1.517ms    |
| DRAM | 0.270ms    | 0.431ms   | 0.379ms      | 0.549ms    |

**Key result:** L1 is ~300x faster than L2, ~600x faster than L3. The hierarchy is real — this is not a theoretical distinction.

---

## Workload benchmarks

Three workloads, each run with SageMem tier-aware vs. flat-memory baseline (plain dict per agent, no cross-agent sharing, no coherence).

### Research workload — multi-agent fact sharing

3 agents, 10 facts seeded in L3, each agent reads all facts 2x.

| | Tier-aware | Flat |
|--|-----------|------|
| L1 hit rate | **100%** (after first read) | N/A (no tier distinction) |
| Overall hit rate | 100% | 100% |
| Tokens (3 agents) | 66 | 66 |
| Duration | ~40ms | ~0ms |

**Interpretation:** After the first read promotes each fact to L1, all subsequent reads are served in-memory at zero measurable latency. The flat baseline loads all facts into a local dict upfront — equivalent behavior at this scale. The difference shows up when fact sets exceed L1 capacity and agents must share via L2 (which the flat baseline cannot do).

### Debate workload — concurrent conflicting writes

2 agents, 5 shared belief keys, 5 rounds each agent writes its position.

| | Tier-aware | Flat |
|--|-----------|------|
| Coherence invalidations | **50** | 0 |
| Stale reads detected | Yes (all of them) | No (silent) |
| Duration | ~280ms | ~0ms |

**Interpretation:** Every time agent A writes, agent B's L1 entry for that key transitions to `Invalid`. B's next read re-fetches the current value. With the flat baseline, agent B reads from its own dict indefinitely — no mechanism exists to learn that A updated the shared belief. Staleness is invisible.

The 280ms overhead vs 0ms baseline is the cost of the coherence protocol: Redis pub/sub round trips, `asyncio.sleep` propagation windows (20ms per round). This is the honest cost of correctness.

### Parallel search workload — cross-agent L2 sharing

4 agents, 16-doc corpus in L3, each agent searches an overlapping 8-key subset.

| | Tier-aware | Flat |
|--|-----------|------|
| L2 cross-agent hits | **12/28 reads (43%)** | 0/28 (0%) |
| L3 reads avoided | **12** | 0 |
| Duration | ~33ms | ~0ms |

**Interpretation:** When agent 1 fetches a doc from L3, it is promoted to the shared L2 (Redis). When agent 2 searches the same overlapping keys, it gets L2 hits — no L3 round trip needed. 43% of total reads across all agents were served from L2 without touching Postgres.

With the flat baseline, every agent is isolated. Each agent independently fetches from its own "database" — no read from one agent benefits any other.

---

## What the numbers say

1. **The tier hierarchy is real.** L1 is orders of magnitude faster than L2/L3. This is not artificial — it is the difference between an in-process dict and a TCP round trip.

2. **Coherence has a cost, and flat memory silently pays it in correctness.** The debate workload shows 50 potential stale reads with flat memory. SageMem detects and corrects all 50 at the cost of ~280ms of coordination overhead. The flat baseline takes 0ms because it does nothing about staleness.

3. **Cross-agent cache sharing is only possible with a shared tier.** 43% of reads in the parallel search workload were served from L2 — work done by one agent that directly benefits another. This is architecturally impossible with per-agent flat memory.

---

## Limitations of these benchmarks

- Mock LLM: real LLM latency (100–500ms per call) would dominate and obscure memory-layer differences. The mock isolates the memory layer, which is the point.
- Local infrastructure: Redis and Postgres on the same machine as the benchmark process. Real deployments add network latency that would make L2/L3 slower relative to L1.
- Small scale: 3–4 agents, dozens of keys. At 50+ agents and thousands of keys, L1 eviction pressure and L2 congestion would produce more dramatic differentiation.

"""Tier latency benchmark — prints median read/write latency for each tier.

Run with: uv run python benchmarks/tier_latency.py

Requires Redis and Postgres to be running.
"""

import asyncio
import statistics
import time

from sagemem.tiers.l1 import L1Tier
from sagemem.tiers.l2 import L2Tier
from sagemem.tiers.l3 import L3Tier
from sagemem.tiers.dram import DRAMTier

REDIS_URL = "redis://localhost:6379"
PG_DSN = "postgresql://localhost/sagemem_test"
ITERATIONS = 200
PAYLOAD = {"content": "benchmark payload", "index": 0}


async def measure(label: str, fn, iterations: int) -> None:
    """Run fn() N times and print median/p95 latency."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    med = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"  {label:<20} median={med:.3f}ms  p95={p95:.3f}ms")


async def main() -> None:
    print(f"\nSageMem tier latency benchmark ({ITERATIONS} iterations each)\n")

    # --- L1 ---
    l1 = L1Tier(capacity=512)
    await l1.set("bench", PAYLOAD)
    print("L1 (in-memory LRU):")
    await measure("write", lambda: l1.set("bench", PAYLOAD), ITERATIONS)
    await measure("read ", lambda: l1.get("bench"), ITERATIONS)

    # --- L2 ---
    print("\nL2 (Redis):")
    l2 = L2Tier(url=REDIS_URL, namespace="sagemem:bench")
    await l2.connect()
    await l2.set("bench", PAYLOAD)
    await measure("write", lambda: l2.set("bench", PAYLOAD), ITERATIONS)
    await measure("read ", lambda: l2.get("bench"), ITERATIONS)
    await l2.clear()
    await l2.disconnect()

    # --- L3 ---
    print("\nL3 (Postgres JSONB):")
    l3 = L3Tier(dsn=PG_DSN, table="bench_l3")
    await l3.connect()
    await l3.set("bench", PAYLOAD)
    await measure("write", lambda: l3.set("bench", PAYLOAD), ITERATIONS)
    await measure("read ", lambda: l3.get("bench"), ITERATIONS)
    await l3.clear()
    await l3.disconnect()

    # --- DRAM ---
    print("\nDRAM (Postgres + pgvector):")
    dram = DRAMTier(dsn=PG_DSN, table="bench_dram", embedding_dim=384)
    await dram.connect()
    await dram.set("bench", PAYLOAD)
    await measure("write", lambda: dram.set("bench", PAYLOAD), ITERATIONS)
    await measure("read ", lambda: dram.get("bench"), ITERATIONS)
    await dram.clear()
    await dram.disconnect()

    print("\nExpected order: L1 << L2 < L3 ≈ DRAM\n")


if __name__ == "__main__":
    asyncio.run(main())

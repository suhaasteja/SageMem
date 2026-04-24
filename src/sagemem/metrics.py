"""Prometheus metrics for SageMem tier operations."""

from prometheus_client import Counter, Histogram

tier_hits = Counter(
    "sagemem_tier_hits_total",
    "Number of cache hits per tier",
    ["tier"],
)

tier_misses = Counter(
    "sagemem_tier_misses_total",
    "Number of cache misses per tier",
    ["tier"],
)

tier_promotions = Counter(
    "sagemem_tier_promotions_total",
    "Number of values promoted to a faster tier on miss",
    ["from_tier", "to_tier"],
)

tier_write_latency = Histogram(
    "sagemem_tier_write_latency_seconds",
    "Write latency per tier",
    ["tier"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

tier_read_latency = Histogram(
    "sagemem_tier_read_latency_seconds",
    "Read latency per tier",
    ["tier"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

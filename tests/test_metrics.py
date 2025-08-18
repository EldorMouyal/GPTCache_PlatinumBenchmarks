import math

from src.metrics import(
    percentile,
    latency_stats,
    hit_miss_counts,
    accuracy_exact,
    aggregate_from_items,
)

def test_percentile_basic():
    assert percentile([], 95) == 0.0
    assert percentile([100], 95) == 100.0
    # Nearest-rank: sorted [10, 20, 30, 40]; p=50 -> ceil(0.5*4)=2 -> index 1 -> 20
    assert percentile([40, 10, 20, 30], 50) == 20.0
    # p=95 on 20 values -> ceil(.95*20)=19 -> index 18 (0-based)
    data = list(range(1, 21))
    assert percentile(data, 95) == 19.0

def test_latency_stats():
    items = [{"latency_ms": x} for x in [10, 20, 30, 40, 50]]
    ls = latency_stats(items)
    assert {"latency_mean_ms", "latency_p95_ms", "latency_p99_ms"} <= set(ls)
    assert math.isclose(ls["latency_mean_ms"], 30.0)
    # p95 of 5 values -> ceil(.95*5)=5 -> value after sort -> 50
    assert ls["latency_p95_ms"] == 50.0
    # p99 -> ceil(.99*5) = 5 -> 50
    assert ls["latency_p99_ms"] == 50.0

def test_hit_miss_counts():
    items = [{"cached": True}, {"cached": False}, {"cached": True}]
    hm = hit_miss_counts(items)
    assert hm["hits"] == 2 and hm["misses"] == 1
    assert hm["hit_rate"] == 2/3

def test_accuracy_exact_normalization():
    items = [
        {"response": "  Hello   WORLD ", "gold": "hello world"},
        {"response": "A", "gold": "B"},
        {"response": "X", "gold": ""},        # ignored (empty gold)
        {"response": "Y"},                    # ignored (no gold)
    ]
    acc = accuracy_exact(items)
    # Of the 2 with non-empty gold, 1 is correct -> 0.5
    assert acc == 0.5


def test_aggregate_from_items_shape():
    items = [
        {"latency_ms": 10, "cached": True,  "response": "ok", "gold": "ok"},
        {"latency_ms": 40, "cached": False, "response": "no", "gold": "ok"},
    ]
    metrics, cache = aggregate_from_items(items)
    assert {"latency_mean_ms","latency_p95_ms","latency_p99_ms","accuracy_exact"} <= set(metrics)
    assert {"hits","misses","hit_rate"} <= set(cache)
    # sanity
    assert metrics["accuracy_exact"] == 0.5
    assert cache["hits"] == 1 and cache["misses"] == 1




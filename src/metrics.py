#!/usr/bin/env python3
"""
metrics.py — Step 6 (Metrics Module) for LLMCache-Bench

Pure-Python, deterministic, import-safe helpers for aggregating:
- Latency statistics: mean, p95, p99 (ms)
- Cache hit/miss counts (+ hit_rate for convenience)
- Exact-match accuracy against `gold` (string) per item

Each `item` is expected to have at least:
  {"latency_ms": float, "cached": bool, "response": str, "gold": str?}

Intended usage in runner.py (later step; do NOT modify runner now):
-------------------------------------------------------------------
    from metrics import aggregate_from_items

    metrics, cache_stats = aggregate_from_items(result["items"])
    result["metrics"] = {
        "latency_mean_ms": metrics["latency_mean_ms"],
        "latency_p95_ms":  metrics["latency_p95_ms"],
        "latency_p99_ms":  metrics["latency_p99_ms"],
        "accuracy_exact":  metrics["accuracy_exact"],
    }
    # If your pipeline also tracks cache stats at item-level:
    result["cache"]["stats"]["hits"]   = cache_stats["hits"]
    result["cache"]["stats"]["misses"] = cache_stats["misses"]
-------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
import math
import statistics


__all__ = [
    "percentile",
    "latency_stats",
    "hit_miss_counts",
    "accuracy_exact",
    "aggregate_from_items",
]


def percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile (inclusive) for a finite iterable.
    Returns 0.0 if empty. Deterministic for ties.
    """
    vals = list(values)
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    rank = max(1, math.ceil((p / 100.0) * len(s)))
    return float(s[rank - 1])


def latency_stats(items: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean, p95, p99 (ms) over `item["latency_ms"]`."""
    lats = [float(it.get("latency_ms", 0.0)) for it in items]
    if not lats:
        return {"latency_mean_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}
    return {
        "latency_mean_ms": float(statistics.mean(lats)),
        "latency_p95_ms": percentile(lats, 95),
        "latency_p99_ms": percentile(lats, 99),
    }


def hit_miss_counts(items: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Count cache hits/misses using `item["cached"]` (bool).
    Returns hits, misses, and hit_rate (0..1; 0 if no items).
    """
    hits = 0
    misses = 0
    for it in items:
        if bool(it.get("cached", False)):
            hits += 1
        else:
            misses += 1
    total = hits + misses
    hit_rate = (hits / total) if total else 0.0
    return {"hits": hits, "misses": misses, "hit_rate": float(hit_rate)}


def _normalize(s: str) -> str:
    """Light normalization for *exact* match: trim + casefold + collapse spaces."""
    s = (s or "").strip().casefold()
    # collapse internal whitespace
    return " ".join(s.split())


def accuracy_exact(items: Iterable[Dict[str, Any]]) -> float:
    """Exact-match accuracy over items that provide a 'gold' string.

    - Compares `_normalize(item["response"]) == _normalize(item["gold"])`.
    - If *no* items contain a non-empty 'gold', returns 0.0 (deterministic).
    """
    have_gold = 0
    correct = 0
    for it in items:
        if "gold" in it and isinstance(it["gold"], str) and it["gold"].strip() != "":
            have_gold += 1
            if _normalize(it.get("response", "")) == _normalize(it.get("gold", "")):
                correct += 1
    if have_gold == 0:
        return 0.0
    return float(correct / have_gold)


def aggregate_from_items(items: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Aggregate all step-6 metrics from a list of result items.

    Returns:
      metrics: {
         "latency_mean_ms", "latency_p95_ms", "latency_p99_ms",
         "accuracy_exact"
      }
      cache_stats: {
         "hits", "misses", "hit_rate"
      }
    """
    lat = latency_stats(items)
    acc = accuracy_exact(items)
    hm = hit_miss_counts(items)

    metrics = {
        "latency_mean_ms": lat["latency_mean_ms"],
        "latency_p95_ms": lat["latency_p95_ms"],
        "latency_p99_ms": lat["latency_p99_ms"],
        "accuracy_exact": acc,
    }
    return metrics, hm

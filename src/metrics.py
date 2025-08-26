# src/metrics.py
"""Lightweight metrics helpers for LLMCache‑Bench.

Exposed pure functions:
- latency_stats(samples) -> {"mean": float, "p95": float, "p99": float}
- hit_rate(hits, total) -> float
- throughput(num_requests, elapsed) -> float
- correctness(expected: list[str], out: str) -> bool

Correctness:
1) Numeric-first: parse model output ("Answer: <num>" or last number in text)
   and expected candidates; compare with a small tolerance.
2) Otherwise, normalized text equality or containment (demo style).
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Optional
import math
import re


# ----------- tiny internal helpers (pure) ------------

_NUM_TOL = 1e-6

def _percentile(sorted_vals: List[float], p: float) -> float:
    """Nearest-rank percentile for 0 < p <= 100. Returns 0.0 on empty input."""
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = math.ceil((p / 100.0) * len(sorted_vals)) - 1  # nearest-rank index
    return float(sorted_vals[max(0, min(k, len(sorted_vals) - 1))])

def _mean(xs: Iterable[float]) -> float:
    s = 0.0
    n = 0
    for v in xs:
        s += float(v)
        n += 1
    return s / n if n else 0.0

def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s([?.!,;:])", r"\1", s)
    return s

def _extract_final_number(text: str) -> Optional[float]:
    """Try 'Answer: <num>' first; else take the last number in the string."""
    m = re.search(r"(?i)answer\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


# -------------------- public API ---------------------

def latency_stats(samples: Iterable[float]) -> Dict[str, float]:
    """Compute mean, p95, p99 (seconds) from an iterable of latency samples (sec)."""
    vals = sorted(float(x) for x in samples)
    return {
        "mean": _mean(vals),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
    }

def hit_rate(hits: int, total: int) -> float:
    """Return hits/total as float; 0.0 when total == 0."""
    return float(hits) / total if total > 0 else 0.0

def throughput(num_requests: int, elapsed: float) -> float:
    """Requests per second; 0.0 when elapsed <= 0."""
    return float(num_requests) / elapsed if elapsed > 0 else 0.0

def correctness(expected: List[str], out: str) -> bool:
    """Demo-aligned correctness.

    1) Numeric path: parse model output and any expected candidate as numbers;
       if both sides yield numbers, accept if |diff| <= _NUM_TOL for any candidate.
    2) Else text path: normalized equality or containment in either direction.
    """
    if out is None:
        return False

    # Numeric-first
    model_num = _extract_final_number(out)
    exp_nums = [_extract_final_number(e) for e in expected or []]
    if model_num is not None and any(e is not None for e in exp_nums):
        return any(e is not None and abs(model_num - e) <= _NUM_TOL for e in exp_nums)

    # Text fallback
    m_norm = _normalize_text(out)
    for e in expected or []:
        n = _normalize_text(str(e))
        if n == m_norm or n in m_norm or m_norm in n:
            return True
    return False

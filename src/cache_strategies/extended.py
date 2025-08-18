"""
Strategy: extended
Behavior: stub for future “smarter” caching (e.g., semantic).
For now: deterministic pseudo-hit based on a configurable hit_rate in [0,1].
  - If hash(prompt) mod 100 < hit_rate*100 -> treat as hit (return canned value).
  - Else miss (store value and return it).
This gives you a controllable hit/miss pattern for experiments before real logic exists.

Config:
  cache:
    mode: extended
    hit_rate: 0.35   # optional, default 0.30

Stats: increments STATS["hits"] / STATS["misses"] accordingly.
"""

from typing import Dict, Any

STATS: Dict[str, int] = {"hits": 0, "misses": 0}
_STATE: Dict[str, Any] = {
    "cache": {},     # prompt -> response string (simulated)
    "config": {},
    "hit_rate": 0.30 # default if not provided
}


def setup_cache(config: Dict[str, Any]) -> None:
    STATS["hits"] = 0
    STATS["misses"] = 0
    _STATE["cache"] = {}
    _STATE["config"] = config or {}
    # pull a hit_rate from config if present
    try:
        _STATE["hit_rate"] = float(_STATE["config"].get("hit_rate", 0.30))
    except Exception:
        _STATE["hit_rate"] = 0.30
    # clamp
    _STATE["hit_rate"] = max(0.0, min(1.0, _STATE["hit_rate"]))


def _should_hit(prompt: str) -> bool:
    # deterministic “randomness”: stable across runs for the same prompt
    bucket = abs(hash(prompt)) % 100
    threshold = int(_STATE["hit_rate"] * 100)
    return bucket < threshold


def query(prompt: str) -> str:
    cache = _STATE["cache"]
    if prompt in cache and _should_hit(prompt):
        STATS["hits"] += 1
        return cache[prompt]
    else:
        STATS["misses"] += 1
        resp = f"[extended-cached-or-missed]:{prompt}"
        cache[prompt] = resp
        return resp

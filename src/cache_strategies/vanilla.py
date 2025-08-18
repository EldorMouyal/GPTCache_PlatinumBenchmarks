"""
Strategy: vanilla
Behavior: simulates a basic exact-match cache (like a naive GPTCache frontend).
         - First time we see a prompt: miss -> store a canned response and return it.
         - Next time: hit -> return the stored response.
Stats: increments STATS["hits"] / STATS["misses"] accordingly.
NOTE: purely a stub; no real GPTCache/FAISS usage here.
"""

from typing import Dict, Any

STATS: Dict[str, int] = {"hits": 0, "misses": 0}
_STATE: Dict[str, Any] = {
    "cache": {},  # prompt -> response string (simulated)
    "config": {},
}


def setup_cache(config: Dict[str, Any]) -> None:
    STATS["hits"] = 0
    STATS["misses"] = 0
    _STATE["cache"] = {}
    _STATE["config"] = config or {}


def query(prompt: str) -> str:
    """
    Simulate exact-match caching.
    """
    cache = _STATE["cache"]
    if prompt in cache:
        STATS["hits"] += 1
        return cache[prompt]
    else:
        STATS["misses"] += 1
        # Simulate that a “backend” produced a response and we cached it.
        resp = f"[vanilla-cached-response]:{prompt}"
        cache[prompt] = resp
        return resp

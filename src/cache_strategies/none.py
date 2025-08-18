"""
Strategy: none
Behavior: no caching (always miss). Returns a dummy string so callers have something to log.
Stats: increments STATS["misses"] on every query.
"""

from typing import Dict, Any

STATS: Dict[str, int] = {"hits": 0, "misses": 0}
_STATE: Dict[str, Any] = {}  # holds config, if needed later


def setup_cache(config: Dict[str, Any]) -> None:
    """Initialize/clear internal state for the 'none' strategy."""
    STATS["hits"] = 0
    STATS["misses"] = 0
    _STATE.clear()
    _STATE["config"] = config or {}


def query(prompt: str) -> str:
    """
    Always a cache miss. Returns a placeholder string.
    (Runner will later decide what to do with a miss.)
    """
    STATS["misses"] += 1
    return f"[none-miss]:{prompt}"

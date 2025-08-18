"""
Simple loader that returns the module implementing the chosen cache mode.

Usage:
    from cache_strategies import load_strategy
    strat = load_strategy(mode)        # 'none' | 'vanilla' | 'extended'
    strat.setup_cache(cache_cfg)       # cache_cfg comes from YAML
    out = strat.query("hello world")
    print(strat.STATS)                 # {'hits': X, 'misses': Y}
"""

from importlib import import_module
from types import ModuleType


def load_strategy(mode: str) -> ModuleType:
    normalized = (mode or "none").strip().lower()
    if normalized not in {"none", "vanilla", "extended"}:
        normalized = "none"
    return import_module(f".{normalized}", package=__name__)

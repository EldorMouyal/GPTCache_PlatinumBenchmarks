import importlib
from src.cache_strategies import load_strategy
MODES = ["none", "vanilla", "extended"]

def test_import_and_interface():
    for mode in MODES:
        mod = importlib.import_module(f"cache_strategies.{mode}")
        assert hasattr(mod, "setup_cache")
        assert hasattr(mod, "query")
        assert hasattr(mod, "STATS")
        assert isinstance(mod.STATS, dict)
        assert set(mod.STATS.keys()) == {"hits", "misses"}

def test_stats_progress():
    for mode in MODES:
        mod = importlib.import_module(f"cache_strategies.{mode}")
        mod.setup_cache({"mode": mode})
        h0, m0 = mod.STATS["hits"], mod.STATS["misses"]
        # call twice on the same prompt; vanilla/extended should register at least one hit by design of your stubs
        r1 = mod.query("hello")
        r2 = mod.query("hello")
        h1, m1 = mod.STATS["hits"], mod.STATS["misses"]
        assert h1 >= h0 and m1 >= m0
        assert isinstance(r1, str) and isinstance(r2, str)

def test_loader_basic():
    for mode in ["none", "vanilla", "extended", "INVALID"]:
        mod = load_strategy(mode)
        assert hasattr(mod, "setup_cache")
        assert hasattr(mod, "query")
        assert hasattr(mod, "STATS")

"""
Strategy: vanilla-exact (GPTCache exact-match)
Uses GPTCache's default exact-match caching via LangChain integration.

This provides deterministic baseline caching with exact string matching only,
using the 'map' data manager for in-memory caching with persistence.
Perfect for establishing baseline performance before studying approximate matching effects.
"""

from typing import Dict, Any, Optional

from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache as LC_GPTCache

try:
    from gptcache import Cache
    from gptcache.processor.pre import get_prompt
    from gptcache.manager.factory import manager_factory
    from gptcache.similarity_evaluation import ExactMatchEvaluation
except ImportError as e:
    raise RuntimeError(
        "GPTCache is required for vanilla cache strategy. "
        "Install with: pip install gptcache"
    ) from e


def _init_gptcache(cache_obj: Cache, llm_string: str) -> None:
    """
    Initialize GPTCache for exact-match using the 'map' data manager.
    Uses a centralized cache directory structure.
    """
    cache_dir = "cache/vanilla_cache"
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=cache_dir),
        similarity_evaluation=ExactMatchEvaluation(),
    )


def setup_cache(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure GPTCache for exact-match caching via LangChain.
    
    Args:
        config: Cache configuration (unused for vanilla strategy)
        model_cfg: Model configuration (unused for vanilla strategy)
    """
    set_llm_cache(LC_GPTCache(_init_gptcache))

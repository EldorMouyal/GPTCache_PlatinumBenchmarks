"""
Strategy: extended (GPTCache loose approximate matching)
Aggressive semantic caching designed to provoke cache-induced errors for reliability testing.

This strategy uses intentionally loose matching criteria to maximize cache hits,
including potentially incorrect ones, allowing study of how aggressive caching 
affects model correctness on PlatinumBench tasks.

Key features for provoking cache errors:
- Very low similarity thresholds (default 0.35)
- Aggressive matching policies that favor speed over accuracy  
- Multiple retrieval strategies to increase false positive rates
- Configurable "looseness" levels for systematic reliability studies
"""

from typing import Dict, Any, Optional

from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache as LC_GPTCache

try:
    from gptcache import Cache, Config
    from gptcache.processor.pre import get_prompt
    from gptcache.manager.factory import manager_factory
    from gptcache.embedding import Onnx
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
except ImportError as e:
    raise RuntimeError(
        "GPTCache with ONNX and FAISS support is required for extended cache strategy. "
        "Install with: pip install gptcache onnxruntime transformers faiss-cpu"
    ) from e


# Preset "looseness" levels for systematic reliability studies
LOOSENESS_PRESETS = {
    "conservative": 0.8,   # High threshold - fewer but safer cache hits  
    "moderate": 0.6,       # Balanced threshold
    "aggressive": 0.35,    # Low threshold - more cache hits, more errors
    "reckless": 0.15,      # Very low threshold - maximum cache hits, maximum errors
}


def _init_gptcache(cache_obj: Cache, llm_string: str) -> None:
    """
    Initialize GPTCache for loose approximate matching using ONNX embeddings and FAISS.
    Configured for aggressive matching that prioritizes cache hits over accuracy.
    """
    # ONNX encoder for sentence embeddings (GPTCache/paraphrase-albert-onnx) 
    encoder = Onnx()
    
    # Use centralized cache directory structure
    cache_dir = "cache/extended_cache"

    # Use sqlite + faiss data manager for vector similarity search
    data_manager = manager_factory(
        "sqlite,faiss", 
        data_dir=cache_dir,
        scalar_params={
            "sql_url": f"sqlite:///./{cache_dir}.db",
            "table_name": "loose_cache"
        },
        vector_params={
            "dimension": encoder.dimension,
            "index_file_path": f"{cache_dir}.index"
        }
    )
    
    # Get similarity threshold from global state (set by setup_cache)
    # Default to aggressive threshold for maximum cache hits
    threshold = getattr(_init_gptcache, '_similarity_threshold', 0.35)
    
    # Initialize cache with loose matching configuration
    cache_obj.init(
        pre_embedding_func=get_prompt,                    # use raw prompt text
        embedding_func=encoder.to_embeddings,            # text -> vector
        data_manager=data_manager,                        # sqlite + faiss storage
        similarity_evaluation=SearchDistanceEvaluation(), # distance-based matching
        config=Config(similarity_threshold=threshold)     # LOW threshold for loose matching
    )


def setup_cache(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure GPTCache for loose approximate matching via LangChain.
    
    Args:
        config: Cache configuration dict with the following options:
            - similarity_threshold: Float threshold (default: 0.35 for aggressive matching)
            - looseness_preset: String preset ("conservative"|"moderate"|"aggressive"|"reckless")
            - looseness_preset overrides similarity_threshold if both provided
        model_cfg: Model configuration (unused for this strategy)
    """
    # Determine similarity threshold with multiple methods for flexibility
    threshold = 0.35  # Default: aggressive threshold
    
    # Method 1: Direct threshold specification
    if "similarity_threshold" in config:
        threshold = float(config["similarity_threshold"])
    
    # Method 2: Preset-based configuration (overrides direct threshold)  
    if "looseness_preset" in config:
        preset = config["looseness_preset"]
        if preset in LOOSENESS_PRESETS:
            threshold = LOOSENESS_PRESETS[preset]
        else:
            available = ", ".join(LOOSENESS_PRESETS.keys())
            raise ValueError(f"Unknown looseness preset '{preset}'. Available: {available}")
    
    # Store threshold in function attribute for _init_gptcache to access
    _init_gptcache._similarity_threshold = threshold
    
    # Wire GPTCache into LangChain with loose matching configuration
    set_llm_cache(LC_GPTCache(_init_gptcache))

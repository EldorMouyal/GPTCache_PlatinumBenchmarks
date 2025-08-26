"""
Strategy: vanilla-approx (GPTCache semantic caching)
Basic semantic caching using GPTCache with ONNX embeddings and FAISS vector store.

This demonstrates the standard GPTCache semantic caching approach as shown in the demos,
with configurable similarity threshold for studying cache hit/miss effects on reliability.
"""

from typing import Dict, Any, Optional
import hashlib

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
        "GPTCache with ONNX and FAISS support is required for vanilla-approx cache strategy. "
        "Install with: pip install gptcache onnxruntime transformers faiss-cpu"
    ) from e


def _hash(s: str) -> str:
    """Generate hash for cache directory naming."""
    return hashlib.sha256(s.encode()).hexdigest()


def _init_gptcache(cache_obj: Cache, llm_string: str) -> None:
    """
    Initialize GPTCache for semantic similarity using ONNX embeddings and FAISS.
    This matches the approach from ollama_gptcache_semantic_demo.py and 
    platinum_gptcache_integration_demo.py.
    """
    # ONNX encoder for sentence embeddings (GPTCache/paraphrase-albert-onnx)
    encoder = Onnx()
    
    # Create unique cache directory per LLM to avoid collisions
    cache_dir = f"vanilla_approx_cache_{_hash(llm_string)}"
    
    # Use sqlite + faiss data manager (standard approach from demos)
    data_manager = manager_factory(
        "sqlite,faiss",
        data_dir=cache_dir,
        scalar_params={
            "sql_url": f"sqlite:///./{cache_dir}.db",
            "table_name": "ollama_cache"
        },
        vector_params={
            "dimension": encoder.dimension,
            "index_file_path": f"{cache_dir}.index"
        }
    )
    
    # Get similarity threshold from global state (set by setup_cache)
    threshold = getattr(_init_gptcache, '_similarity_threshold', 0.75)
    
    # Initialize cache with semantic similarity configuration
    cache_obj.init(
        pre_embedding_func=get_prompt,                    # use raw prompt text
        embedding_func=encoder.to_embeddings,            # text -> vector
        data_manager=data_manager,                        # sqlite + faiss storage
        similarity_evaluation=SearchDistanceEvaluation(), # distance-based matching
        config=Config(similarity_threshold=threshold)     # configurable threshold
    )


def setup_cache(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure GPTCache for semantic caching via LangChain.
    
    Args:
        config: Cache configuration dict, expects 'similarity_threshold' key
        model_cfg: Model configuration (unused for this strategy)
    """
    # Extract similarity threshold from config (default to 0.75 like demos)
    threshold = config.get("similarity_threshold", 0.75)
    
    # Store threshold in function attribute for _init_gptcache to access
    _init_gptcache._similarity_threshold = threshold
    
    # Wire GPTCache into LangChain
    set_llm_cache(LC_GPTCache(_init_gptcache))
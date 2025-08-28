"""
Strategy: custom-semantic (Configurable GPTCache semantic caching)
Semantic caching with multiple embedding models for systematic experimentation.

This strategy supports multiple high-performance embedding models with configurable
similarity thresholds. Designed for systematic optimization and comparison across
embedding models and similarity thresholds.
"""

from typing import Dict, Any, Optional

from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache as LC_GPTCache

try:
    from gptcache import Cache, Config
    from gptcache.processor.pre import get_prompt
    from gptcache.manager.factory import manager_factory
    from gptcache.embedding import Onnx, Huggingface, SBERT
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
except ImportError as e:
    raise RuntimeError(
        "GPTCache with embedding support is required for custom-semantic cache strategy. "
        "Install with: pip install gptcache onnxruntime transformers faiss-cpu sentence-transformers"
    ) from e


# Embedding model configurations
EMBEDDING_MODELS = {
    "onnx": {
        "class": Onnx,
        "model": "GPTCache/paraphrase-albert-onnx",
        "description": "ONNX paraphrase-albert baseline"
    },
    "bge_small": {
        "class": Huggingface,
        "model": "BAAI/bge-small-en-v1.5",
        "description": "BGE small English v1.5 - high performance, compact"
    },
    "e5_base": {
        "class": Huggingface,
        "model": "intfloat/e5-base-v2",
        "description": "E5 base v2 - strong general-purpose embeddings"
    },
    "mpnet": {
        "class": SBERT,
        "model": "all-mpnet-base-v2",
        "description": "MPNet base - sentence-transformers reference model"
    }
}


def _init_custom_cache(cache_obj: Cache, llm_string: str) -> None:
    """
    Initialize GPTCache with configurable embedding model.
    
    Uses modern embedding models for semantic similarity matching.
    """
    # Get configuration from global state (set by setup_cache)
    embedding_model_key = getattr(_init_custom_cache, '_embedding_model', 'onnx')
    similarity_threshold = getattr(_init_custom_cache, '_similarity_threshold', 0.8)
    
    # Initialize embedding model
    if embedding_model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model: {embedding_model_key}. "
                        f"Available models: {list(EMBEDDING_MODELS.keys())}")
    
    model_config = EMBEDDING_MODELS[embedding_model_key]
    
    # Create encoder instance
    if model_config["class"] == Onnx:
        encoder = model_config["class"]()  # ONNX uses default model
    else:
        encoder = model_config["class"](model=model_config["model"])
    
    # Use embedding-specific cache directory
    cache_dir = f"cache/custom_semantic_{embedding_model_key}_cache"
    
    # Configure data manager with sqlite + faiss
    data_manager = manager_factory(
        "sqlite,faiss",
        data_dir=cache_dir,
        scalar_params={
            "sql_url": f"sqlite:///./{cache_dir}.db",
            "table_name": "custom_cache"
        },
        vector_params={
            "dimension": encoder.dimension,
            "index_file_path": f"{cache_dir}.index"
        }
    )
    
    # Initialize cache with standard configuration
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=encoder.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=similarity_threshold)
    )


def setup_cache(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure custom semantic caching with modern embedding models.
    
    Args:
        config: Cache configuration dict with the following options:
            - similarity_threshold: Float threshold (default: 0.8)
            - embedding_model: String model key from EMBEDDING_MODELS (default: 'onnx')
        model_cfg: Model configuration (unused for this strategy)
    """
    # Extract configuration parameters
    similarity_threshold = config.get("similarity_threshold", 0.8)
    embedding_model = config.get("embedding_model", "onnx")
    
    # Validate embedding model
    if embedding_model not in EMBEDDING_MODELS:
        available = list(EMBEDDING_MODELS.keys())
        raise ValueError(f"Unknown embedding model '{embedding_model}'. Available: {available}")
    
    # Store configuration in function attributes for _init_custom_cache to access
    _init_custom_cache._similarity_threshold = similarity_threshold
    _init_custom_cache._embedding_model = embedding_model
    
    # Wire custom GPTCache into LangChain
    set_llm_cache(LC_GPTCache(_init_custom_cache))
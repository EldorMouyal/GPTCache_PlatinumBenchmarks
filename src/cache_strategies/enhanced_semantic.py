"""
Strategy: enhanced-semantic (Advanced GPTCache semantic caching)
Modern semantic caching with multiple embedding models and conservative quality validation.

This strategy supports multiple high-performance embedding models and includes
conservative quality filtering to reduce bad cache hits while maximizing cache hit rates.
Designed for systematic optimization across embedding models and similarity thresholds.
"""

from typing import Dict, Any, Optional
import re
import math

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
        "GPTCache with embedding support is required for enhanced-semantic cache strategy. "
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


def _validate_response_quality(prompt: str, response: str, similarity_score: float, threshold: float) -> bool:
    """
    Conservative response quality validation to reduce bad cache hits.
    
    Only filters obvious quality issues - prioritizes safety over aggressive filtering.
    Better to cache a potentially imperfect response than reject a good one.
    
    Args:
        prompt: Input query
        response: Generated response
        similarity_score: Embedding similarity score
        threshold: Configured similarity threshold
        
    Returns:
        True if response should be cached, False if it should be filtered out
    """
    # Filter completely empty or whitespace-only responses
    if not response or not response.strip():
        return False
    
    # Filter responses that are obviously truncated (end mid-word or mid-sentence)
    if response.endswith(('...', '…')) or re.search(r'\w+$', response.strip()) is None:
        return False
    
    # Conservative length-based filtering - only reject extreme mismatches
    # Allow 10x difference to be very permissive
    prompt_length = len(prompt.split())
    response_length = len(response.split())
    
    if response_length > 0 and prompt_length > 0:
        length_ratio = max(response_length, prompt_length) / min(response_length, prompt_length)
        if length_ratio > 10:  # Very permissive threshold
            return False
    
    # Embedding confidence check - reject matches very close to threshold
    # Only filter if similarity is within 2% of threshold (very conservative)
    confidence_buffer = 0.02
    if similarity_score < threshold + confidence_buffer:
        return False
    
    # Pass all other cases - be conservative about rejecting responses
    return True


def _init_enhanced_cache(cache_obj: Cache, llm_string: str) -> None:
    """
    Initialize GPTCache with configurable embedding model and quality validation.
    
    Uses modern embedding models with conservative quality filtering to improve
    cache hit accuracy while maintaining high recall.
    """
    # Get configuration from global state (set by setup_cache)
    embedding_model_key = getattr(_init_enhanced_cache, '_embedding_model', 'onnx')
    similarity_threshold = getattr(_init_enhanced_cache, '_similarity_threshold', 0.8)
    enable_quality_validation = getattr(_init_enhanced_cache, '_enable_quality_validation', True)
    
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
    cache_dir = f"cache/enhanced_semantic_{embedding_model_key}_cache"
    
    # Configure data manager with sqlite + faiss
    data_manager = manager_factory(
        "sqlite,faiss",
        data_dir=cache_dir,
        scalar_params={
            "sql_url": f"sqlite:///./{cache_dir}.db",
            "table_name": "enhanced_cache"
        },
        vector_params={
            "dimension": encoder.dimension,
            "index_file_path": f"{cache_dir}.index"
        }
    )
    
    # Custom similarity evaluation with optional quality validation
    class EnhancedSimilarityEvaluation(SearchDistanceEvaluation):
        def evaluation(self, src_dict, cache_dict, **kwargs):
            # Get base similarity score
            similarity_score = super().evaluation(src_dict, cache_dict, **kwargs)
            
            # Apply quality validation if enabled
            if enable_quality_validation:
                prompt = src_dict.get('question', '')
                response = cache_dict.get('answer', '')
                
                if not _validate_response_quality(prompt, response, similarity_score, similarity_threshold):
                    return 0.0  # Force cache miss by returning very low similarity
            
            return similarity_score
    
    # Initialize cache with enhanced configuration
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=encoder.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=EnhancedSimilarityEvaluation(),
        config=Config(similarity_threshold=similarity_threshold)
    )


def setup_cache(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure enhanced semantic caching with modern embedding models.
    
    Args:
        config: Cache configuration dict with the following options:
            - similarity_threshold: Float threshold (default: 0.8)
            - embedding_model: String model key from EMBEDDING_MODELS (default: 'onnx')
            - enable_quality_validation: Bool to enable/disable quality filtering (default: True)
        model_cfg: Model configuration (unused for this strategy)
    """
    # Extract configuration parameters
    similarity_threshold = config.get("similarity_threshold", 0.8)
    embedding_model = config.get("embedding_model", "onnx")
    enable_quality_validation = config.get("enable_quality_validation", True)
    
    # Validate embedding model
    if embedding_model not in EMBEDDING_MODELS:
        available = list(EMBEDDING_MODELS.keys())
        raise ValueError(f"Unknown embedding model '{embedding_model}'. Available: {available}")
    
    # Store configuration in function attributes for _init_enhanced_cache to access
    _init_enhanced_cache._similarity_threshold = similarity_threshold
    _init_enhanced_cache._embedding_model = embedding_model
    _init_enhanced_cache._enable_quality_validation = enable_quality_validation
    
    # Wire enhanced GPTCache into LangChain
    set_llm_cache(LC_GPTCache(_init_enhanced_cache))
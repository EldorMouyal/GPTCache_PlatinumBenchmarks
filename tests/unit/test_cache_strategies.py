# tests/unit/test_cache_strategies.py
import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from src.cache_strategies import none, vanilla_exact, vanilla_approx


class TestNoneStrategy:
    """Test cache_strategies/none.py - disables caching entirely."""
    
    def test_setup_cache_basic(self):
        """Test that setup_cache calls set_llm_cache(None)."""
        with patch('src.cache_strategies.none.set_llm_cache') as mock_set_cache:
            none.setup_cache({})
            mock_set_cache.assert_called_once_with(None)
    
    def test_setup_cache_with_config(self):
        """Test that setup_cache ignores config and still disables cache."""
        with patch('src.cache_strategies.none.set_llm_cache') as mock_set_cache:
            config = {"mode": "none", "some_param": "ignored"}
            none.setup_cache(config)
            mock_set_cache.assert_called_once_with(None)


class TestVanillaExactStrategy:
    """Test cache_strategies/vanilla_exact.py - GPTCache exact matching."""
    
    def test_hash_function(self):
        """Test _hash function produces consistent results."""
        from src.cache_strategies.vanilla_exact import _hash
        
        # Same input should produce same hash
        assert _hash("test") == _hash("test")
        # Different inputs should produce different hashes
        assert _hash("test1") != _hash("test2")
        # Hash should be hex string
        hash_result = _hash("test")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length
    
    @patch('src.cache_strategies.vanilla_exact.set_llm_cache')
    @patch('src.cache_strategies.vanilla_exact.LC_GPTCache')
    def test_setup_cache_basic(self, mock_lc_gptcache, mock_set_cache):
        """Test basic setup_cache functionality."""
        vanilla_exact.setup_cache({}, {})
        
        # Should create LC_GPTCache with init function
        mock_lc_gptcache.assert_called_once()
        # Should set the cache
        mock_set_cache.assert_called_once()
    
    @patch('src.cache_strategies.vanilla_exact.manager_factory')
    @patch('src.cache_strategies.vanilla_exact.get_prompt')
    @patch('src.cache_strategies.vanilla_exact.ExactMatchEvaluation')
    def test_init_gptcache_components(self, mock_exact_match, mock_get_prompt, mock_manager_factory):
        """Test _init_gptcache initializes with correct components."""
        from src.cache_strategies.vanilla_exact import _init_gptcache
        
        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager_factory.return_value = mock_manager
        mock_evaluator = Mock()
        mock_exact_match.return_value = mock_evaluator
        
        _init_gptcache(mock_cache, "test_llm_string")
        
        # Should create manager with map type
        mock_manager_factory.assert_called_once_with(manager="map", data_dir=StringMatching(r"vanilla_cache_.*"))
        
        # Should initialize cache with correct components
        mock_cache.init.assert_called_once_with(
            pre_embedding_func=mock_get_prompt,
            data_manager=mock_manager,
            similarity_evaluation=mock_evaluator
        )


class TestVanillaApproxStrategy:
    """Test cache_strategies/vanilla_approx.py - GPTCache semantic caching."""
    
    def test_hash_function(self):
        """Test _hash function produces consistent results."""
        from src.cache_strategies.vanilla_approx import _hash
        
        # Same input should produce same hash
        assert _hash("test") == _hash("test")
        # Different inputs should produce different hashes
        assert _hash("test1") != _hash("test2")
        # Hash should be hex string
        hash_result = _hash("test")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length
    
    @patch('src.cache_strategies.vanilla_approx.set_llm_cache')
    @patch('src.cache_strategies.vanilla_approx.LC_GPTCache')
    def test_setup_cache_basic(self, mock_lc_gptcache, mock_set_cache):
        """Test basic setup_cache functionality."""
        config = {"similarity_threshold": 0.8}
        vanilla_approx.setup_cache(config, {})
        
        # Should create LC_GPTCache with init function
        mock_lc_gptcache.assert_called_once()
        # Should set the cache
        mock_set_cache.assert_called_once()
    
    def test_setup_cache_threshold_config(self):
        """Test that similarity threshold is extracted from config."""
        from src.cache_strategies.vanilla_approx import _init_gptcache
        
        # Test default threshold
        vanilla_approx.setup_cache({}, {})
        assert getattr(_init_gptcache, '_similarity_threshold', None) == 0.75
        
        # Test custom threshold
        vanilla_approx.setup_cache({"similarity_threshold": 0.9}, {})
        assert getattr(_init_gptcache, '_similarity_threshold', None) == 0.9
    
    @patch('src.cache_strategies.vanilla_approx.manager_factory')
    @patch('src.cache_strategies.vanilla_approx.get_prompt')
    @patch('src.cache_strategies.vanilla_approx.SearchDistanceEvaluation')
    @patch('src.cache_strategies.vanilla_approx.Config')
    @patch('src.cache_strategies.vanilla_approx.Onnx')
    def test_init_gptcache_components(self, mock_onnx, mock_config, mock_search_eval, mock_get_prompt, mock_manager_factory):
        """Test _init_gptcache initializes with correct semantic components."""
        from src.cache_strategies.vanilla_approx import _init_gptcache
        
        # Setup mocks
        mock_encoder = Mock()
        mock_encoder.dimension = 384
        mock_onnx.return_value = mock_encoder
        
        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager_factory.return_value = mock_manager
        mock_evaluator = Mock()
        mock_search_eval.return_value = mock_evaluator
        mock_cfg = Mock()
        mock_config.return_value = mock_cfg
        
        # Set threshold attribute
        _init_gptcache._similarity_threshold = 0.8
        
        _init_gptcache(mock_cache, "test_llm_string")
        
        # Should create ONNX encoder
        mock_onnx.assert_called_once()
        
        # Should create manager with sqlite,faiss type
        mock_manager_factory.assert_called_once_with(
            "sqlite,faiss",
            data_dir=StringMatching(r"vanilla_approx_cache_.*"),
            scalar_params={
                "sql_url": StringMatching(r"sqlite:///.*vanilla_approx_cache_.*.db"),
                "table_name": "ollama_cache"
            },
            vector_params={
                "dimension": 384,
                "index_file_path": StringMatching(r".*vanilla_approx_cache_.*.index")
            }
        )
        
        # Should create config with threshold
        mock_config.assert_called_once_with(similarity_threshold=0.8)
        
        # Should initialize cache with semantic components
        mock_cache.init.assert_called_once_with(
            pre_embedding_func=mock_get_prompt,
            embedding_func=mock_encoder.to_embeddings,
            data_manager=mock_manager,
            similarity_evaluation=mock_evaluator,
            config=mock_cfg
        )


class TestImportRequirements:
    """Test that all strategies handle import failures gracefully."""
    
    def test_vanilla_exact_import_error_handling(self):
        """Test that vanilla_exact has proper import error handling in place."""
        # Since the module has already imported successfully, we test the structure
        # The actual import error would happen at module load time
        import src.cache_strategies.vanilla_exact as ve_module
        
        # Verify the module has the import handling structure
        import inspect
        source = inspect.getsource(ve_module)
        assert "ImportError" in source, "Module should handle ImportError"
        assert "RuntimeError" in source, "Module should raise RuntimeError on missing deps"
        assert "GPTCache is required" in source, "Module should have helpful error message"
    
    def test_vanilla_approx_import_error_handling(self):
        """Test that vanilla_approx has proper import error handling in place."""
        # Since the module has already imported successfully, we test the structure
        import src.cache_strategies.vanilla_approx as va_module
        
        # Verify the module has the import handling structure  
        import inspect
        source = inspect.getsource(va_module)
        assert "ImportError" in source, "Module should handle ImportError"
        assert "RuntimeError" in source, "Module should raise RuntimeError on missing deps"
        assert "GPTCache with ONNX and FAISS support is required" in source, "Module should have helpful error message"


# Custom string matcher for test assertions
class StringMatching:
    def __init__(self, pattern):
        import re
        self.pattern = re.compile(pattern)
    
    def __eq__(self, other):
        return bool(self.pattern.search(str(other)))
    
    def __repr__(self):
        return f"<StringMatching {self.pattern.pattern}>"
# tests/unit/test_extended_strategy.py
import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from src.cache_strategies import extended


class TestExtendedStrategy:
    """Test cache_strategies/extended.py - loose approximate matching."""
    
    def test_hash_function(self):
        """Test _hash function produces consistent results."""
        # Same input should produce same hash
        assert extended._hash("test") == extended._hash("test")
        # Different inputs should produce different hashes
        assert extended._hash("test1") != extended._hash("test2")
        # Hash should be hex string
        hash_result = extended._hash("test")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length
    
    def test_looseness_presets(self):
        """Test that looseness presets are properly defined."""
        assert "conservative" in extended.LOOSENESS_PRESETS
        assert "moderate" in extended.LOOSENESS_PRESETS
        assert "aggressive" in extended.LOOSENESS_PRESETS
        assert "reckless" in extended.LOOSENESS_PRESETS
        
        # Conservative should have highest threshold, reckless lowest
        assert extended.LOOSENESS_PRESETS["conservative"] > extended.LOOSENESS_PRESETS["moderate"]
        assert extended.LOOSENESS_PRESETS["moderate"] > extended.LOOSENESS_PRESETS["aggressive"] 
        assert extended.LOOSENESS_PRESETS["aggressive"] > extended.LOOSENESS_PRESETS["reckless"]
    
    @patch.object(extended, 'set_llm_cache')
    @patch.object(extended, 'LC_GPTCache')
    def test_setup_cache_with_direct_threshold(self, mock_lc_gptcache, mock_set_cache):
        """Test setup_cache with direct similarity_threshold."""
        config = {"similarity_threshold": 0.2}
        extended.setup_cache(config)
        
        # Check threshold was stored
        assert getattr(extended._init_gptcache, '_similarity_threshold', None) == 0.2
        
        # Verify LangChain cache setup
        mock_lc_gptcache.assert_called_once()
        mock_set_cache.assert_called_once()
    
    @patch.object(extended, 'set_llm_cache')
    @patch.object(extended, 'LC_GPTCache')
    def test_setup_cache_with_preset(self, mock_lc_gptcache, mock_set_cache):
        """Test setup_cache with looseness_preset (should override threshold)."""
        config = {
            "similarity_threshold": 0.9,  # This should be overridden
            "looseness_preset": "reckless"
        }
        extended.setup_cache(config)
        
        # Preset should override direct threshold
        expected = extended.LOOSENESS_PRESETS["reckless"]
        assert getattr(extended._init_gptcache, '_similarity_threshold', None) == expected
        
        # Verify LangChain cache setup
        mock_lc_gptcache.assert_called_once()
        mock_set_cache.assert_called_once()
    
    @patch.object(extended, 'set_llm_cache')
    @patch.object(extended, 'LC_GPTCache')
    def test_setup_cache_with_invalid_preset(self, mock_lc_gptcache, mock_set_cache):
        """Test setup_cache raises error for invalid preset."""
        config = {"looseness_preset": "invalid_preset"}
        
        with pytest.raises(ValueError, match="Unknown looseness preset 'invalid_preset'"):
            extended.setup_cache(config)
    
    @patch.object(extended, 'set_llm_cache')
    @patch.object(extended, 'LC_GPTCache')
    def test_setup_cache_defaults(self, mock_lc_gptcache, mock_set_cache):
        """Test setup_cache uses default aggressive threshold when no config provided."""
        extended.setup_cache({})
        
        # Should use default aggressive threshold
        assert getattr(extended._init_gptcache, '_similarity_threshold', None) == 0.35
        
        # Verify LangChain cache setup
        mock_lc_gptcache.assert_called_once()
        mock_set_cache.assert_called_once()
    
    @patch.object(extended, 'Onnx')
    @patch.object(extended, 'manager_factory')
    @patch.object(extended, 'get_prompt')
    @patch.object(extended, 'SearchDistanceEvaluation')
    @patch.object(extended, 'Config')
    def test_init_gptcache_components(self, mock_config, mock_search_eval, mock_get_prompt, mock_manager_factory, mock_onnx):
        """Test _init_gptcache initializes with correct loose matching components."""
        # Setup mocks
        mock_encoder = Mock()
        mock_encoder.dimension = 384
        mock_encoder.to_embeddings = Mock()
        mock_onnx.return_value = mock_encoder
        
        mock_manager = Mock()
        mock_manager_factory.return_value = mock_manager
        
        mock_evaluator = Mock()
        mock_search_eval.return_value = mock_evaluator
        
        mock_cfg = Mock()
        mock_config.return_value = mock_cfg
        
        mock_cache = Mock()
        
        # Set aggressive threshold and initialize
        extended._init_gptcache._similarity_threshold = 0.25
        extended._init_gptcache(mock_cache, "test_llm")
        
        # Verify ONNX encoder creation
        mock_onnx.assert_called_once()
        
        # Verify manager creation with sqlite+faiss
        mock_manager_factory.assert_called_once()
        args, kwargs = mock_manager_factory.call_args
        assert args[0] == "sqlite,faiss"
        assert "data_dir" in kwargs
        assert kwargs["data_dir"].startswith("extended_loose_cache_")
        assert "scalar_params" in kwargs
        assert "vector_params" in kwargs
        
        # Verify table name is "loose_cache"
        scalar_params = kwargs["scalar_params"]
        assert scalar_params["table_name"] == "loose_cache"
        
        # Verify vector params use encoder dimension
        vector_params = kwargs["vector_params"]
        assert vector_params["dimension"] == 384
        
        # Verify config creation with LOW threshold (aggressive matching)
        mock_config.assert_called_once_with(similarity_threshold=0.25)
        
        # Verify cache initialization with loose matching components
        mock_cache.init.assert_called_once_with(
            pre_embedding_func=mock_get_prompt,
            embedding_func=mock_encoder.to_embeddings,
            data_manager=mock_manager,
            similarity_evaluation=mock_evaluator,
            config=mock_cfg
        )
    
    def test_all_presets_are_valid_thresholds(self):
        """Test that all preset values are valid similarity thresholds (0.0 to 1.0)."""
        for preset_name, threshold in extended.LOOSENESS_PRESETS.items():
            assert 0.0 <= threshold <= 1.0, f"Preset {preset_name} has invalid threshold {threshold}"
            assert isinstance(threshold, (int, float)), f"Preset {preset_name} threshold must be numeric"


class TestExtendedImportHandling:
    """Test that extended strategy handles import failures gracefully."""
    
    def test_import_error_handling_structure(self):
        """Test that extended.py has proper import error handling in place."""
        # Verify the module has the import handling structure
        import inspect
        source = inspect.getsource(extended)
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
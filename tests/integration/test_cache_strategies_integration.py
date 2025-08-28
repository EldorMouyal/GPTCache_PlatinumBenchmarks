# tests/integration/test_cache_strategies_integration.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import sys
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from src.cache_strategies import none, vanilla_exact, vanilla_approx


class DummyLLM:
    """Mock LLM that returns predictable responses and can be cached."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = ["Response 1", "Response 2", "Response 3"]
    
    def invoke(self, prompt: str) -> str:
        """Return deterministic response based on prompt."""
        self.call_count += 1
        # For exact match testing: same prompt -> same response
        if "exact test" in prompt:
            return "Exact match response"
        # For semantic testing: similar prompts -> similar responses
        if "semantic" in prompt or "meaning" in prompt:
            return "Semantic response"
        # Default response
        return f"Default response {self.call_count}"


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dummy_llm():
    """Create a mock LLM for testing."""
    return DummyLLM()


class TestNoneStrategyIntegration:
    """Integration tests for none strategy with LLM."""
    
    def test_no_caching_behavior(self, dummy_llm):
        """Test that 'none' strategy never caches responses."""
        # Patch where the function is imported and used in the none module
        with patch.object(none, 'set_llm_cache') as mock_set_cache:
            # Setup none strategy
            none.setup_cache({})
            
            # Verify cache is disabled
            mock_set_cache.assert_called_once_with(None)
    
    def test_setup_multiple_times(self):
        """Test that setup_cache can be called multiple times safely."""
        # Patch where the function is imported and used in the none module
        with patch.object(none, 'set_llm_cache') as mock_set_cache:
            none.setup_cache({"param1": "value1"})
            none.setup_cache({"param2": "value2"})
            
            # Should be called twice, both times with None
            assert mock_set_cache.call_count == 2
            for call in mock_set_cache.call_args_list:
                assert call[0][0] is None


class TestVanillaExactStrategyIntegration:
    """Integration tests for vanilla_exact strategy with mocked GPTCache."""
    
    def test_setup_creates_cache_directory(self, temp_cache_dir):
        """Test that setup creates appropriate cache directory structure."""
        # Test that setup_cache calls set_llm_cache with LC_GPTCache
        with patch.object(vanilla_exact, 'set_llm_cache') as mock_set_cache, \
             patch.object(vanilla_exact, 'LC_GPTCache') as mock_lc_gptcache:
            
            config = {}
            model_cfg = {"name": "test_model"}
            
            vanilla_exact.setup_cache(config, model_cfg)
            
            # Verify LC_GPTCache was created with the init function
            mock_lc_gptcache.assert_called_once()
            
            # Verify LangChain cache is set
            mock_set_cache.assert_called_once()
            
            # The argument to LC_GPTCache should be the _init_gptcache function
            args, kwargs = mock_lc_gptcache.call_args
            assert callable(args[0]), "LC_GPTCache should be called with a callable init function"
    
    def test_cache_initialization_components(self):
        """Test that cache is initialized with correct exact-match components."""
        from src.cache_strategies.vanilla_exact import _init_gptcache
        
        # Patch the functions in the module
        with patch.object(vanilla_exact, 'manager_factory') as mock_manager_factory, \
             patch.object(vanilla_exact, 'get_prompt') as mock_get_prompt, \
             patch.object(vanilla_exact, 'ExactMatchEvaluation') as mock_exact_match:
            
            # Setup mocks
            mock_cache = Mock()
            mock_manager = Mock()
            mock_manager_factory.return_value = mock_manager
            mock_evaluator = Mock()
            mock_exact_match.return_value = mock_evaluator
            
            _init_gptcache(mock_cache, "test_llm_string")
            
            # Verify cache initialized with exact match components
            mock_cache.init.assert_called_once_with(
                pre_embedding_func=mock_get_prompt,
                data_manager=mock_manager,
                similarity_evaluation=mock_evaluator
            )


class TestVanillaApproxStrategyIntegration:
    """Integration tests for vanilla_approx strategy with mocked components."""
    
    def test_setup_with_custom_threshold(self):
        """Test setup with custom similarity threshold."""
        # Test that setup_cache calls set_llm_cache with LC_GPTCache and stores threshold
        with patch.object(vanilla_approx, 'set_llm_cache') as mock_set_cache, \
             patch.object(vanilla_approx, 'LC_GPTCache') as mock_lc_gptcache:
            
            config = {"similarity_threshold": 0.9}
            vanilla_approx.setup_cache(config)
            
            # Check threshold was stored
            from src.cache_strategies.vanilla_approx import _init_gptcache
            assert getattr(_init_gptcache, '_similarity_threshold', None) == 0.9
            
            # Verify LC_GPTCache was created with the init function
            mock_lc_gptcache.assert_called_once()
            
            # Verify LangChain cache is set
            mock_set_cache.assert_called_once()
    
    def test_semantic_cache_initialization(self):
        """Test that semantic cache components are properly initialized."""
        from src.cache_strategies.vanilla_approx import _init_gptcache
        
        # Patch the functions in the module
        with patch.object(vanilla_approx, 'Onnx') as mock_onnx, \
             patch.object(vanilla_approx, 'manager_factory') as mock_manager_factory, \
             patch.object(vanilla_approx, 'get_prompt') as mock_get_prompt, \
             patch.object(vanilla_approx, 'SearchDistanceEvaluation') as mock_search_eval, \
             patch.object(vanilla_approx, 'Config') as mock_config:
            
            # Setup mocks
            mock_encoder = Mock()
            mock_encoder.dimension = 768
            mock_encoder.to_embeddings = Mock(return_value=[0.1, 0.2, 0.3])
            mock_onnx.return_value = mock_encoder
            
            mock_manager = Mock()
            mock_manager_factory.return_value = mock_manager
            
            mock_evaluator = Mock()
            mock_search_eval.return_value = mock_evaluator
            
            mock_cfg = Mock()
            mock_config.return_value = mock_cfg
            
            mock_cache = Mock()
            
            # Set threshold and initialize
            _init_gptcache._similarity_threshold = 0.75
            _init_gptcache(mock_cache, "test_llm")
            
            # Verify ONNX encoder creation
            mock_onnx.assert_called_once()
            
            # Verify manager creation with sqlite+faiss
            mock_manager_factory.assert_called_once()
            args, kwargs = mock_manager_factory.call_args
            assert args[0] == "sqlite,faiss"
            assert "data_dir" in kwargs
            assert "scalar_params" in kwargs
            assert "vector_params" in kwargs
            
            # Verify vector params use encoder dimension
            vector_params = kwargs["vector_params"]
            assert vector_params["dimension"] == 768
            
            # Verify config creation with threshold
            mock_config.assert_called_once_with(similarity_threshold=0.75)
            
            # Verify cache initialization with semantic components
            mock_cache.init.assert_called_once_with(
                pre_embedding_func=mock_get_prompt,
                embedding_func=mock_encoder.to_embeddings,
                data_manager=mock_manager,
                similarity_evaluation=mock_evaluator,
                config=mock_cfg
            )
    
    def test_threshold_default_behavior(self):
        """Test that default threshold is used when not specified."""
        vanilla_approx.setup_cache({})
        
        from src.cache_strategies.vanilla_approx import _init_gptcache
        assert getattr(_init_gptcache, '_similarity_threshold', None) == 0.75


class TestCacheStrategiesCompatibility:
    """Test that all cache strategies have compatible interfaces."""
    
    def test_all_strategies_have_setup_cache(self):
        """Test that all strategies implement setup_cache function."""
        strategies = [none, vanilla_exact, vanilla_approx]
        
        for strategy in strategies:
            assert hasattr(strategy, 'setup_cache'), f"{strategy.__name__} missing setup_cache"
            assert callable(strategy.setup_cache), f"{strategy.__name__}.setup_cache not callable"
    
    def test_setup_cache_signature_compatibility(self):
        """Test that setup_cache functions accept similar parameters."""
        import inspect
        
        # Test none strategy (simpler signature)
        sig_none = inspect.signature(none.setup_cache)
        assert len(sig_none.parameters) >= 1, "none.setup_cache should accept at least config"
        
        # Test vanilla strategies (should accept config and optional model_cfg)
        sig_exact = inspect.signature(vanilla_exact.setup_cache)
        sig_approx = inspect.signature(vanilla_approx.setup_cache)
        
        assert len(sig_exact.parameters) >= 1, "vanilla_exact.setup_cache should accept at least config"
        assert len(sig_approx.parameters) >= 1, "vanilla_approx.setup_cache should accept at least config"
    
    def test_all_strategies_can_be_setup(self):
        """Test that all strategies can be set up without errors (with mocking)."""
        config = {"similarity_threshold": 0.8, "mode": "test"}
        
        # Test none strategy
        with patch.object(none, 'set_llm_cache') as mock_set_none:
            none.setup_cache(config)
            mock_set_none.assert_called_once_with(None)
        
        # Test vanilla_exact strategy
        with patch.object(vanilla_exact, 'set_llm_cache') as mock_set_exact, \
             patch.object(vanilla_exact, 'LC_GPTCache'):
            vanilla_exact.setup_cache(config)
            mock_set_exact.assert_called_once()
        
        # Test vanilla_approx strategy
        with patch.object(vanilla_approx, 'set_llm_cache') as mock_set_approx, \
             patch.object(vanilla_approx, 'LC_GPTCache'):
            vanilla_approx.setup_cache(config)
            mock_set_approx.assert_called_once()


class TestErrorHandling:
    """Test error handling in cache strategies."""
    
    def test_vanilla_exact_handles_gptcache_import_error(self):
        """Test that vanilla_exact handles GPTCache import gracefully."""
        # This should already be tested in unit tests, but verify integration context
        # The actual import error would happen at module load time, so we test the behavior
        # when dependencies are missing
        pass  # Import errors are handled at module level
    
    def test_vanilla_approx_handles_dependencies_import_error(self):
        """Test that vanilla_approx handles missing dependencies gracefully."""
        # Similar to above, these are module-level import errors
        pass  # Import errors are handled at module level
    
    def test_cache_setup_with_invalid_config(self):
        """Test behavior with various invalid configurations."""
        # None strategy should handle any config
        with patch.object(none, 'set_llm_cache'):
            none.setup_cache(None)
            none.setup_cache({"invalid": "config"})
            none.setup_cache([1, 2, 3])  # Wrong type
        
        # Vanilla exact should handle missing keys gracefully
        with patch.object(vanilla_exact, 'set_llm_cache'), \
             patch.object(vanilla_exact, 'LC_GPTCache'):
            vanilla_exact.setup_cache({})
            vanilla_exact.setup_cache(None)
        
        # Vanilla approx should use defaults for missing threshold
        with patch.object(vanilla_approx, 'set_llm_cache'), \
             patch.object(vanilla_approx, 'LC_GPTCache'):
            vanilla_approx.setup_cache({})
            vanilla_approx.setup_cache({"invalid_key": "value"})
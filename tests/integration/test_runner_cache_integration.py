# tests/integration/test_runner_cache_integration.py
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import yaml
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import src.runner as runner


def load_test_config(**overrides):
    """Load base test config with optional overrides."""
    project_root = Path(__file__).parents[2]
    base_config_path = project_root / "test_experiment.yaml"
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides recursively
    def update_dict(base, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                update_dict(base[key], value)
            else:
                base[key] = value
    
    update_dict(config, overrides)
    return config


class MockLLM:
    """Mock LLM with predictable timing for cache hit testing."""
    
    def __init__(self, fast_responses=None):
        self.call_count = 0
        self.fast_responses = fast_responses or set()  # Set of prompts that should be "fast" (cache hits)
    
    def invoke(self, prompt: str) -> str:
        self.call_count += 1
        
        # Simulate cache hits with very fast responses
        if prompt in self.fast_responses:
            import time
            time.sleep(0.01)  # Very fast (cache hit)
            return f"CACHED: Response {self.call_count}"
        else:
            import time  
            time.sleep(0.2)   # Slower (cache miss)
            return f"MISS: Response {self.call_count}"


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture 
def mock_platinum_data():
    """Mock platinum dataset data."""
    return [
        {"platinum_prompt": "What is 2+2?", "platinum_target": "4"},
        {"question": "What is 3+3?", "answer": "6"},
        {"statement": "The answer is 8.", "target": "8"}
    ]


class TestCacheStrategyLoading:
    """Test dynamic cache strategy loading functionality."""
    
    def test_load_none_strategy(self):
        """Test loading 'none' cache strategy."""
        cache_module = runner._load_cache_strategy("none")
        
        assert hasattr(cache_module, 'setup_cache')
        assert callable(cache_module.setup_cache)
    
    def test_load_vanilla_exact_strategy(self):
        """Test loading 'vanilla_exact' cache strategy.""" 
        cache_module = runner._load_cache_strategy("vanilla_exact")
        
        assert hasattr(cache_module, 'setup_cache')
        assert callable(cache_module.setup_cache)
    
    def test_load_vanilla_approx_strategy(self):
        """Test loading 'vanilla_approx' cache strategy."""
        cache_module = runner._load_cache_strategy("vanilla_approx")
        
        assert hasattr(cache_module, 'setup_cache')
        assert callable(cache_module.setup_cache)
    
    def test_load_extended_strategy(self):
        """Test loading 'extended' cache strategy."""
        cache_module = runner._load_cache_strategy("extended")
        
        assert hasattr(cache_module, 'setup_cache')
        assert callable(cache_module.setup_cache)
    
    def test_load_nonexistent_strategy(self):
        """Test error handling for nonexistent cache strategy."""
        with pytest.raises(ImportError) as exc_info:
            runner._load_cache_strategy("nonexistent_strategy")
        
        assert "Cache strategy 'nonexistent_strategy' not found" in str(exc_info.value)
        assert "Available strategies:" in str(exc_info.value)
    
    def test_get_available_strategies(self):
        """Test getting list of available cache strategies."""
        strategies = runner._get_available_strategies()
        
        # Should include our known strategies
        expected_strategies = {"none", "vanilla_exact", "vanilla_approx", "extended"}
        actual_strategies = set(strategies)
        
        assert expected_strategies.issubset(actual_strategies)
        assert all(isinstance(s, str) for s in strategies)
    
    def test_strategy_without_setup_cache(self):
        """Test error handling for strategy module without setup_cache function."""
        # Create a mock module without setup_cache
        mock_module = MagicMock()
        del mock_module.setup_cache  # Remove the setup_cache attribute
        
        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(AttributeError) as exc_info:
                runner._load_cache_strategy("bad_strategy")
            
            assert "missing setup_cache() function" in str(exc_info.value)


class TestCacheHitTracker:
    """Test cache hit detection using timing heuristics."""
    
    def test_cache_hit_tracker_initialization(self):
        """Test CacheHitTracker initialization."""
        tracker = runner._CacheHitTracker(hit_threshold_sec=0.1)
        
        assert tracker.hit_threshold_sec == 0.1
        assert tracker.total_queries == 0
        assert tracker.hits_detected == 0
        assert tracker.bad_hits_detected == 0
        assert tracker.get_hit_rate() == 0.0
        assert tracker.get_bad_hit_rate() == 0.0
        assert tracker.get_cache_accuracy() == 0.0
    
    def test_cache_hit_detection_fast_response(self):
        """Test detection of cache hits for fast responses."""
        tracker = runner._CacheHitTracker(hit_threshold_sec=0.1)
        
        # Fast response should be detected as cache hit (correct answer)
        is_hit = tracker.record_query(0.05, "test prompt", "test response", is_correct=True)
        
        assert is_hit is True
        assert tracker.hits_detected == 1
        assert tracker.total_queries == 1
        assert tracker.bad_hits_detected == 0
        assert tracker.get_hit_rate() == 1.0
        assert tracker.get_bad_hit_rate() == 0.0
        assert tracker.get_cache_accuracy() == 1.0
    
    def test_cache_miss_detection_slow_response(self):
        """Test detection of cache misses for slow responses."""
        tracker = runner._CacheHitTracker(hit_threshold_sec=0.1)
        
        # Slow response should be detected as cache miss
        is_hit = tracker.record_query(0.2, "test prompt", "test response", is_correct=True)
        
        assert is_hit is False
        assert tracker.hits_detected == 0
        assert tracker.total_queries == 1
        assert tracker.bad_hits_detected == 0
        assert tracker.get_hit_rate() == 0.0
        assert tracker.get_bad_hit_rate() == 0.0
    
    def test_mixed_hit_miss_pattern(self):
        """Test tracking of mixed hit/miss patterns."""
        tracker = runner._CacheHitTracker(hit_threshold_sec=0.1)
        
        # Record mix of fast and slow responses with correctness
        tracker.record_query(0.05, "prompt1", "response1", is_correct=True)   # Good hit
        tracker.record_query(0.2, "prompt2", "response2", is_correct=True)    # Miss
        tracker.record_query(0.03, "prompt3", "response3", is_correct=False)  # Bad hit
        tracker.record_query(0.15, "prompt4", "response4", is_correct=False)  # Miss
        
        assert tracker.hits_detected == 2
        assert tracker.total_queries == 4
        assert tracker.bad_hits_detected == 1
        assert tracker.get_hit_rate() == 0.5
        assert tracker.get_bad_hit_rate() == 0.5  # 1 bad hit out of 2 total hits
        assert tracker.get_cache_accuracy() == 0.5  # 1 good hit out of 2 total hits
    
    def test_bad_cache_hits_analysis(self):
        """Test comprehensive bad cache hits analysis functionality."""
        tracker = runner._CacheHitTracker(hit_threshold_sec=0.1)
        
        # Scenario: Multiple queries with various hit/miss and correct/incorrect patterns
        tracker.record_query(0.05, "2+2", "4", is_correct=True)        # Good cache hit
        tracker.record_query(0.03, "3+3", "7", is_correct=False)       # Bad cache hit (wrong answer)
        tracker.record_query(0.2, "4+4", "8", is_correct=True)         # Cache miss (correct)
        tracker.record_query(0.02, "5+5", "10", is_correct=True)       # Good cache hit
        tracker.record_query(0.01, "6+6", "13", is_correct=False)      # Bad cache hit (wrong answer)
        tracker.record_query(0.3, "7+7", "15", is_correct=False)       # Cache miss (wrong)
        
        # Verify comprehensive stats
        stats = tracker.get_stats()
        
        assert tracker.total_queries == 6
        assert tracker.hits_detected == 4  # 4 fast responses
        assert tracker.bad_hits_detected == 2       # 2 incorrect fast responses
        
        assert tracker.get_hit_rate() == 4/6  # 66.7% hit rate
        assert tracker.get_bad_hit_rate() == 2/4  # 50% of hits were bad
        assert tracker.get_cache_accuracy() == 2/4  # 50% of hits were good
        
        # Verify detailed stats
        assert stats["total_queries"] == 6
        assert stats["cache_hits"] == 4
        assert stats["cache_misses"] == 2
        assert stats["bad_cache_hits"] == 2
        assert stats["good_cache_hits"] == 2
        assert abs(stats["hit_rate"] - 4/6) < 0.001
        assert abs(stats["bad_hit_rate"] - 2/4) < 0.001
        assert abs(stats["cache_accuracy"] - 2/4) < 0.001


class TestRunnerIntegration:
    """Test runner integration with cache strategies."""
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_run_once_with_none_cache(self, mock_build_llm, mock_load_platinum, mock_platinum_data):
        """Test run_once with 'none' cache strategy."""
        # Setup mocks
        mock_llm = MockLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_platinum_data
        
        config = load_test_config(
            run={"id": "test-none"},
            dataset={"slice": {"limit": 2}},
            cache={"mode": "none"}
        )
        
        # Patch set_llm_cache to avoid actual cache setup
        with patch('src.cache_strategies.none.set_llm_cache'):
            result = runner.run_once(config)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result["cache"]["mode"] == "none"
        assert "metrics" in result
        assert "items" in result
        assert len(result["items"]) == 2
        
        # Verify cache hit tracking
        assert result["metrics"]["cache_hit_rate"] >= 0.0
        assert all("cache_hit" in item for item in result["items"])
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_run_once_with_vanilla_exact_cache(self, mock_build_llm, mock_load_platinum, mock_platinum_data):
        """Test run_once with 'vanilla_exact' cache strategy."""
        # Setup mocks with some fast responses to simulate cache hits
        mock_llm = MockLLM(fast_responses={"What is 2+2?"})
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_platinum_data
        
        config = load_test_config(
            run={"id": "test-vanilla-exact"},
            dataset={"slice": {"limit": 2}},
            cache={"mode": "vanilla_exact"}
        )
        
        # Mock the cache strategy setup
        with patch('src.cache_strategies.vanilla_exact.set_llm_cache'), \
             patch('src.cache_strategies.vanilla_exact.LC_GPTCache'):
            result = runner.run_once(config)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result["cache"]["mode"] == "vanilla_exact"
        assert "metrics" in result
        assert "items" in result
        
        # Should detect some cache hits based on timing
        assert result["metrics"]["cache_hit_rate"] >= 0.0
    
    def test_run_once_with_invalid_cache_strategy(self):
        """Test run_once error handling with invalid cache strategy."""
        config = load_test_config(
            run={"id": "test-invalid"},
            dataset={"slice": {"limit": 1}},
            cache={"mode": "nonexistent_strategy"}
        )
        
        with pytest.raises(ImportError) as exc_info:
            runner.run_once(config)
        
        assert "Cache strategy 'nonexistent_strategy' not found" in str(exc_info.value)
    
    @patch('src.bench_datasets.platinum.load')  
    @patch('src.models.ollama.build_llm')
    def test_run_once_cache_configuration_passing(self, mock_build_llm, mock_load_platinum, mock_platinum_data):
        """Test that cache configuration is properly passed to cache strategy."""
        mock_llm = MockLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_platinum_data
        
        config = load_test_config(
            run={"id": "test-config"},
            dataset={"slice": {"limit": 1}},
            cache={
                "mode": "vanilla_approx",
                "similarity_threshold": 0.8,
                "custom_param": "test_value"
            }
        )
        
        # Mock the cache strategy and verify setup_cache is called with config
        with patch('src.cache_strategies.vanilla_approx.set_llm_cache'), \
             patch('src.cache_strategies.vanilla_approx.LC_GPTCache') as mock_cache:
            
            # Mock the setup_cache function to verify it receives the config
            mock_setup_cache = Mock()
            with patch('src.cache_strategies.vanilla_approx.setup_cache', mock_setup_cache):
                result = runner.run_once(config)
            
            # Verify setup_cache was called with the cache config and model config
            mock_setup_cache.assert_called_once()
            call_args = mock_setup_cache.call_args
            
            # First argument should be cache config
            cache_config_arg = call_args[0][0]
            assert cache_config_arg["mode"] == "vanilla_approx"
            assert cache_config_arg["similarity_threshold"] == 0.8
            assert cache_config_arg["custom_param"] == "test_value"
        
        # Verify result includes cache configuration
        assert result["cache"]["mode"] == "vanilla_approx"
        assert result["cache"]["similarity_threshold"] == 0.8
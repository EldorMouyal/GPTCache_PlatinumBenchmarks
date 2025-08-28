# tests/integration/test_end2end_smoke.py
import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import src.runner as runner


class DummyLLM:
    """Dummy LLM that returns predictable responses."""
    
    def __init__(self):
        self.call_count = 0
        
    def invoke(self, prompt: str) -> str:
        self.call_count += 1
        
        # Return deterministic responses based on prompt content
        if "2+2" in prompt or "2 + 2" in prompt:
            return "4"
        elif "3+3" in prompt or "3 + 3" in prompt:
            return "6"
        elif "answer is 8" in prompt.lower():
            return "8"
        else:
            return f"Answer: {self.call_count}"


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiment files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_dataset_fixture():
    """Mock dataset that returns small test data."""
    return [
        {"platinum_prompt": "What is 2+2?", "platinum_target": "4"},
        {"question": "What is 3+3?", "answer": "6"},  
        {"statement": "The answer is 8.", "target": "8"}
    ]


def create_test_config(temp_dir: Path, cache_mode: str, **cache_params):
    """Create a test experiment configuration file based on test_experiment.yaml."""
    # Load base test configuration
    project_root = Path(__file__).parents[2]  # Go up from tests/integration/ to project root
    base_config_path = project_root / "test_experiment.yaml"
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override specific settings for this test
    config["run"]["id"] = f"smoke-test-{cache_mode}"
    config["run"]["notes"] = f"End-to-end smoke test with {cache_mode} cache"
    config["cache"]["mode"] = cache_mode
    config["cache"].update(cache_params)  # Add any additional cache parameters
    config["output"]["dir"] = str(temp_dir / "results")
    
    # Write the modified config to temp directory
    config_path = temp_dir / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, default_flow_style=False))
    return config_path


class TestEndToEndSmoke:
    """End-to-end smoke tests for runner with different cache strategies."""
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_e2e_none_cache_strategy(self, mock_build_llm, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """End-to-end test with 'none' cache strategy."""
        # Setup mocks
        mock_llm = DummyLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_dataset_fixture
        
        # Create test config
        config_path = create_test_config(temp_experiment_dir, "none")
        
        # Mock the cache strategy
        with patch('src.cache_strategies.none.set_llm_cache'):
            # Run experiment
            runner.main(str(config_path))
        
        # Verify output file was created
        results_dir = temp_experiment_dir / "results"
        assert results_dir.exists()
        
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1
        
        # Load and verify results
        result = json.loads(result_files[0].read_text())
        
        assert result["run_id"] == "smoke-test-none"
        assert result["cache"]["mode"] == "none"
        assert result["metrics"]["cache_hit_rate"] == 0.0  # No caching
        assert result["metrics"]["correctness"] == 1.0     # All answers correct
        assert len(result["items"]) == 3
        
        # Verify all items have cache_hit=False for none strategy
        for item in result["items"]:
            assert item["cache_hit"] is False
            assert item["correct"] is True
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_e2e_vanilla_exact_cache_strategy(self, mock_build_llm, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """End-to-end test with 'vanilla_exact' cache strategy."""
        # Setup mocks
        mock_llm = DummyLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_dataset_fixture
        
        # Create test config
        config_path = create_test_config(temp_experiment_dir, "vanilla_exact")
        
        # Mock the cache strategy
        with patch('src.cache_strategies.vanilla_exact.set_llm_cache'), \
             patch('src.cache_strategies.vanilla_exact.LC_GPTCache'):
            
            # Run experiment
            runner.main(str(config_path))
        
        # Verify output file was created
        results_dir = temp_experiment_dir / "results"
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1
        
        # Load and verify results
        result = json.loads(result_files[0].read_text())
        
        assert result["run_id"] == "smoke-test-vanilla_exact"
        assert result["cache"]["mode"] == "vanilla_exact"
        assert result["metrics"]["cache_hit_rate"] >= 0.0
        assert result["metrics"]["correctness"] == 1.0
        assert len(result["items"]) == 3
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_e2e_vanilla_approx_cache_strategy(self, mock_build_llm, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """End-to-end test with 'vanilla_approx' cache strategy."""
        # Setup mocks
        mock_llm = DummyLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_dataset_fixture
        
        # Create test config with similarity threshold
        config_path = create_test_config(
            temp_experiment_dir, 
            "vanilla_approx", 
            similarity_threshold=0.8
        )
        
        # Mock the cache strategy
        with patch('src.cache_strategies.vanilla_approx.set_llm_cache'), \
             patch('src.cache_strategies.vanilla_approx.LC_GPTCache'):
            
            # Run experiment
            runner.main(str(config_path))
        
        # Verify output file was created and contains expected data
        results_dir = temp_experiment_dir / "results"
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1
        
        result = json.loads(result_files[0].read_text())
        
        assert result["run_id"] == "smoke-test-vanilla_approx"
        assert result["cache"]["mode"] == "vanilla_approx"
        assert result["cache"]["similarity_threshold"] == 0.8
        assert result["metrics"]["cache_hit_rate"] >= 0.0
        assert result["metrics"]["correctness"] == 1.0
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_e2e_extended_cache_strategy(self, mock_build_llm, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """End-to-end test with 'extended' cache strategy using preset."""
        # Setup mocks
        mock_llm = DummyLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_dataset_fixture
        
        # Create test config with looseness preset
        config_path = create_test_config(
            temp_experiment_dir,
            "extended",
            looseness_preset="aggressive"
        )
        
        # Mock the cache strategy
        with patch('src.cache_strategies.extended.set_llm_cache'), \
             patch('src.cache_strategies.extended.LC_GPTCache'):
            
            # Run experiment
            runner.main(str(config_path))
        
        # Verify output file was created and contains expected data
        results_dir = temp_experiment_dir / "results"
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1
        
        result = json.loads(result_files[0].read_text())
        
        assert result["run_id"] == "smoke-test-extended"
        assert result["cache"]["mode"] == "extended"
        assert result["cache"]["looseness_preset"] == "aggressive"
        assert result["metrics"]["cache_hit_rate"] >= 0.0
        assert result["metrics"]["correctness"] == 1.0
    
    def test_e2e_invalid_cache_strategy(self, temp_experiment_dir):
        """End-to-end test with invalid cache strategy should fail gracefully."""
        # Create test config with invalid cache strategy
        config_path = create_test_config(temp_experiment_dir, "nonexistent_strategy")
        
        # Should raise ImportError for invalid strategy
        with pytest.raises(ImportError) as exc_info:
            runner.main(str(config_path))
        
        assert "Cache strategy 'nonexistent_strategy' not found" in str(exc_info.value)
        assert "Available strategies:" in str(exc_info.value)
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_e2e_custom_output_directory(self, mock_build_llm, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """Test end-to-end with custom output directory and filename pattern."""
        # Setup mocks
        mock_llm = DummyLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_dataset_fixture
        
        # Load base test config and customize for this test
        project_root = Path(__file__).parents[2]
        base_config_path = project_root / "test_experiment.yaml"
        
        with open(base_config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Override specific settings for custom output test
        custom_config["run"]["id"] = "custom-output-test"
        custom_config["dataset"]["slice"]["limit"] = 1  # Single item test
        custom_config["cache"]["mode"] = "none"
        custom_config["output"]["dir"] = str(temp_experiment_dir / "custom_results")
        custom_config["output"]["filename_pattern"] = "experiment_{run_id}_results.json"
        
        config_path = temp_experiment_dir / "custom_experiment.yaml"
        config_path.write_text(yaml.safe_dump(custom_config))
        
        # Mock cache strategy
        with patch('src.cache_strategies.none.set_llm_cache'):
            runner.main(str(config_path))
        
        # Verify custom output location
        custom_results_dir = temp_experiment_dir / "custom_results"
        assert custom_results_dir.exists()
        
        expected_filename = "experiment_custom-output-test_results.json"
        result_file = custom_results_dir / expected_filename
        assert result_file.exists()
        
        # Verify content
        result = json.loads(result_file.read_text())
        assert result["run_id"] == "custom-output-test"


class TestRunnerRobustness:
    """Test runner robustness and error handling."""
    
    def test_empty_dataset_handling(self, temp_experiment_dir):
        """Test runner behavior with empty dataset."""
        config_path = create_test_config(temp_experiment_dir, "none")
        
        # Mock empty dataset
        with patch('src.bench_datasets.platinum.load', return_value=[]), \
             patch('src.cache_strategies.none.set_llm_cache'):
            
            runner.main(str(config_path))
        
        # Should still create result file with empty items
        results_dir = temp_experiment_dir / "results" 
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1
        
        result = json.loads(result_files[0].read_text())
        assert len(result["items"]) == 0
        assert result["metrics"]["correctness"] == 0.0
        assert result["metrics"]["cache_hit_rate"] == 0.0
    
    def test_missing_config_file(self):
        """Test runner behavior with missing config file."""
        with pytest.raises(FileNotFoundError):
            runner.main("/nonexistent/path/to/config.yaml")
    
    @patch('src.bench_datasets.platinum.load')
    @patch('src.models.ollama.build_llm')
    def test_cache_strategy_setup_failure(self, mock_build_llm, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """Test runner behavior when cache strategy setup fails."""
        # Setup basic mocks
        mock_llm = DummyLLM()
        mock_build_llm.return_value = mock_llm
        mock_load_platinum.return_value = mock_dataset_fixture
        
        config_path = create_test_config(temp_experiment_dir, "vanilla_exact")
        
        # Mock cache strategy setup to raise an exception
        with patch('src.cache_strategies.vanilla_exact.setup_cache', side_effect=RuntimeError("Cache setup failed")):
            with pytest.raises(RuntimeError) as exc_info:
                runner.main(str(config_path))
            
            assert "Cache setup failed" in str(exc_info.value)
    
    @patch('src.bench_datasets.platinum.load')
    def test_model_build_failure(self, mock_load_platinum, temp_experiment_dir, mock_dataset_fixture):
        """Test runner behavior when model building fails."""
        mock_load_platinum.return_value = mock_dataset_fixture
        config_path = create_test_config(temp_experiment_dir, "none")
        
        # Mock model building to fail
        with patch('src.models.ollama.build_llm', side_effect=RuntimeError("Model build failed")), \
             patch('src.cache_strategies.none.set_llm_cache'):
            
            with pytest.raises(RuntimeError) as exc_info:
                runner.main(str(config_path))
            
            assert "Model build failed" in str(exc_info.value)
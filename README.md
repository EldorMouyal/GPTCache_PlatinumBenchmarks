# LLMCache-Bench

A comprehensive benchmarking framework for evaluating **semantic caching strategies** in Large Language Models (LLMs). Test different caching approaches, measure performance gains, and optimize your LLM applications.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Multiple Cache Strategies**: None, exact matching, semantic similarity, and advanced configurations
- **GPU Acceleration**: Free Tesla T4 GPU via Kaggle notebooks for 10-20x speedup  
- **Flexible Deployment**: Local development, Docker containers, or cloud environments
- **Comprehensive Metrics**: Latency, hit rates, correctness, and cache quality analysis
- **Real Datasets**: PlatinumBench with GSM8K, HotpotQA, SingleQ, and more
- **Dynamic Loading**: Add custom cache strategies without code changes

## 📋 Quick Start

> 🚀 **Want 10-20x faster performance?** Skip to the [GPU Acceleration (Free!)](#-gpu-acceleration-free) section to use Kaggle's free Tesla T4 GPUs instead of local CPU processing.

### Option 1: Docker (Recommended)

#### Initial Setup

```bash
# Clone and enter the repository
git clone https://github.com/EldorMouyal/GPTCache_PlatinumBenchmarks.git
cd GPTCache_PlatinumBenchmarks

# Start the Ollama server
docker-compose up -d ollama

# Pull required model (this may take a few minutes on CPU)
docker-compose exec ollama ollama pull gemma2:2b

# Verify Ollama is working
docker-compose exec ollama ollama list

# Build the benchmark application
docker build -t llmcache_bench .

# 💡 TIP: For faster performance, consider using free GPU acceleration (see GPU section below)
```

#### Running Experiments

> **Note**: The `--network host` flag allows the Docker container to access your local Ollama server. This is required when using the Docker Ollama setup.

**Linux/macOS/Git Bash:**
```bash
# Quick smoke test (5 examples, ~2 minutes)
docker run --network host -v $(pwd)/results:/app/results llmcache_bench \
  python scripts/run.py --config experiments/smoke_test.yaml

# Full default experiment (200 examples, ~15 minutes)  
docker run --network host -v $(pwd)/results:/app/results llmcache_bench \
  python scripts/run.py --config experiments/experiment.yaml
```

**Windows PowerShell:**
```powershell
# Quick smoke test (5 examples, ~2 minutes)
docker run --network host -v ${PWD}/results:/app/results llmcache_bench python scripts/run.py --config experiments/smoke_test.yaml

# Full default experiment (200 examples, ~15 minutes)
docker run --network host -v ${PWD}/results:/app/results llmcache_bench python scripts/run.py --config experiments/experiment.yaml
```

**Windows CMD:**
```cmd
# Quick smoke test (5 examples, ~2 minutes)
docker run --network host -v %cd%/results:/app/results llmcache_bench python scripts/run.py --config experiments/smoke_test.yaml

# Full default experiment (200 examples, ~15 minutes)
docker run --network host -v %cd%/results:/app/results llmcache_bench python scripts/run.py --config experiments/experiment.yaml
```

**Custom Configuration (any platform):**
```bash
# Copy and edit config file
cp experiments/experiment.yaml my_config.yaml

# Run with custom config (adjust volume syntax for your platform)
docker run -v $(pwd)/my_config.yaml:/app/experiments/my_config.yaml \
           -v $(pwd)/results:/app/results llmcache-bench \
  python scripts/run.py --config experiments/my_config.yaml
```

#### Using Remote GPU Server

```bash
# Connect to remote Ollama server (e.g., Kaggle GPU)
docker run -e OLLAMA_BASE_URL=https://your-ngrok-url.com \
           -v $(pwd)/results:/app/results llmcache-bench \
  python scripts/run.py --config experiments/experiment.yaml
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start local Ollama
ollama serve

# Pull a model
ollama pull gemma2:9b

# Run experiments
python scripts/run.py
```

## ⚡ GPU Acceleration (Free!)

Speed up experiments by 10-20x using free GPU resources:

1. **Upload to Kaggle**: Upload `kaggle_ollama_server.ipynb` to Kaggle
2. **Enable GPU**: Set accelerator to "GPU T4 x2" in notebook settings  
3. **Get ngrok token**: Sign up at [ngrok.com](https://ngrok.com) for free
4. **Run notebook**: Execute all cells, copy the public URL
5. **Switch config**: `python scripts/setup_gpu_config.py https://your-ngrok-url.com`
6. **Run experiments**: `python scripts/run.py` (same commands, GPU speed!)

```bash
# Check GPU connection
python scripts/check_gpu_status.py

# Switch back to local
python scripts/setup_gpu_config.py http://localhost:11434
```

## 🧪 Usage Examples

### Basic Experiment

```bash
# Run with default config (GSM8K + HotpotQA + SingleQ)
python scripts/run.py

# Use custom config
python scripts/run.py --config experiments/my_experiment.yaml

# Quick smoke test (5 examples)
python scripts/run.py --config experiments/smoke_test.yaml
```

### Docker Advanced Usage

```bash
# Development mode with live code changes
docker run --network host \
           -v $(pwd)/src:/app/src \
           -v $(pwd)/experiments:/app/experiments \
           -v $(pwd)/results:/app/results \
           llmcache_bench python scripts/run.py

# Run with environment variables (for remote GPU servers)
docker run -e OLLAMA_BASE_URL=https://your-gpu-server.com \
           -v $(pwd)/results:/app/results \
           llmcache_bench python scripts/run.py

# Note: --network host is needed for local Ollama connectivity
# Omit it only when using remote Ollama servers via OLLAMA_BASE_URL
```

### Testing

```bash
# Run all tests
python -m pytest -q

# Unit tests only
python -m pytest tests/unit/ -q

# Integration tests (requires running Ollama)
python -m pytest tests/integration/ -q

# Verbose output
python -m pytest -xvs
```

## 🧪 How to Benchmark

### Benchmarking Methodology

Our benchmarking framework evaluates cache strategies across multiple dimensions to provide comprehensive performance insights.

#### Workload Profiles

**Dataset-Based Evaluation**: Uses PlatinumBench subsets to simulate real-world scenarios:
- **Repetitive Patterns**: GSM8K math problems with similar question structures (tests cache hit potential)
- **Diverse Content**: HotpotQA multi-hop reasoning (tests semantic similarity matching)
- **Mixed Complexity**: SingleQ, MMLU-Math, Winograd (comprehensive evaluation)

**Cache Hit Scenarios**:
- **High Hit Rate**: Similar questions within same subset (e.g., "What is 2+3?" vs "Calculate 5+7")
- **Low Hit Rate**: Diverse question types across subsets
- **Cache Overhead**: Novel prompts unlikely to be cached (measures baseline impact)

#### Metrics Collected

**Performance Metrics**:
- `latency_mean_sec`: Average response time per query
- `latency_p95_sec`: 95th percentile latency (captures worst-case performance)
- `throughput_qps`: Queries processed per second

**Cache Effectiveness**:
- `cache_hit_rate`: Percentage of queries served from cache
- `bad_cache_hit_rate`: Percentage of cache hits that were incorrect
- `cache_accuracy`: Ratio of correct cache hits to total cache hits
- `cache_effectiveness`: Overall cache performance improvement

**Quality Metrics**:
- `correctness`: Percentage of responses matching expected answers
- `reliability_degradation`: Quality impact from caching
- `correctness_without_bad_hits`: Correctness excluding bad cache hits

#### Running Benchmarks

```bash
# Basic benchmark run
python scripts/run.py

# Compare multiple strategies
python scripts/run.py --config experiments/baseline_comparison.yaml

# Parameter sweeping (similarity thresholds)
python scripts/run.py --config experiments/threshold_sweep.yaml

# Generate analysis plots
python scripts/plot_embedding_comparison.py

# Aggregate results across experiments
python scripts/aggregate.py
```

#### Automated Testing Integration

Benchmarks integrate with the testing suite:

```bash
# Run performance tests with pytest-benchmark
python -m pytest tests/integration/ --benchmark-only

# Full test suite including benchmarks
python -m pytest tests/ -v
```

#### Result Interpretation

**Summary Results**: Located in `results/tables/summary.csv`

| Metric | Meaning | Good Values |
|--------|---------|-------------|
| `correctness` | Response accuracy | >0.9 (90%+) |
| `cache_hit_rate` | Cache utilization | 0.2-0.7 (20-70%) |
| `cache_accuracy` | Cache quality | >0.6 (60%+) |
| `latency_p95_sec` | Response time | <5.0 seconds |
| `cache_effectiveness` | Performance gain | >0.1 (10%+) |

**Example Results**:
```csv
run_id,correctness,cache_hit_rate,cache_accuracy,latency_mean_sec,cache_effectiveness
custom_mpnet_095,0.925,0.21,0.714,4.63,0.15
none,0.965,0.105,1.0,4.47,0.105
```

**Individual Results**: Detailed JSON files following `schema/result.schema.json` structure:
```json
{
  "run": {"id": "experiment_name", "seed": 42},
  "metrics": {
    "latency_mean_sec": 4.63,
    "cache_hit_rate": 0.21,
    "correctness": 0.925
  },
  "items": [{"prompt": "...", "response": "...", "latency_ms": 4630, "cached": true}]
}
```

## ⚙️ Configuration

Experiments are configured via YAML files in `experiments/`. Key sections:

```yaml
run:
  id: "my_experiment"
  seed: 42

model:
  provider: "ollama"
  name: "gemma3:4b"
  base_url: "http://localhost:11434"  # or GPU URL
  params:
    temperature: 0.0
    max_tokens: 256

dataset:
  name: "platinum-bench"
  subsets: ["gsm8k", "hotpotqa", "singleq"]
  split: "test"
  slice:
    start: 0
    limit: 25

cache:
  mode: "vanilla_approx"  # none, vanilla_exact, vanilla_approx, extended
  similarity_threshold: 0.95
```

## 📊 Cache Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `none` | No caching | Baseline comparison |
| `vanilla_exact` | Exact string matching | Identical queries |
| `vanilla_approx` | Semantic similarity (FAISS) | Similar questions |
| `extended` | Advanced similarity with presets | Fine-tuned caching |

Add custom strategies by creating new files in `src/cache_strategies/` - they're loaded dynamically!

## 🏗️ Architecture

```
src/
├── runner.py              # Main orchestrator with dynamic loading
├── models/ollama.py       # LangChain Ollama integration
├── cache_strategies/      # Pluggable caching strategies
├── bench_datasets/        # Dataset loaders (PlatinumBench)
├── metrics.py            # Performance and correctness metrics
└── utils/                # Timing, I/O utilities

scripts/
├── run.py                # Execute experiments
├── setup_gpu_config.py   # Switch between local/GPU Ollama
├── check_gpu_status.py   # Verify connections
└── clear_cache.py        # Cache management

experiments/
└── experiment.yaml       # Default experiment config

results/
├── raw/                  # JSON results per experiment
├── figures/              # Generated plots
└── tables/               # Summary statistics
```

## 🔧 Development

### Adding Cache Strategies

1. Create `src/cache_strategies/my_strategy.py`
2. Implement `setup_cache(config, model_cfg)` function
3. Set `cache.mode: "my_strategy"` in experiment config
4. Run experiments - strategy loads automatically!

### Environment Variables

- `OLLAMA_BASE_URL`: Override config file base_url (great for Docker)

### Cache Management

```bash
# Clear all caches
python scripts/clear_cache.py

# Preview what would be deleted
python scripts/clear_cache.py --dry-run

# Force clear without prompt
python scripts/clear_cache.py --yes
```

## 📈 Results

Experiments generate:
- **Raw results**: JSON files with detailed metrics (`results/raw/`)
- **Aggregated data**: Summary statistics and comparisons
- **Visualizations**: Hit rate vs correctness plots, latency analysis
- **Schema validation**: Results follow `schema/result.schema.json`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-cache`  
3. Add tests for new functionality
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [GPTCache](https://github.com/zilliztech/GPTCache) for caching infrastructure
- [PlatinumBench](https://huggingface.co/datasets/platinum-bench) for evaluation datasets
- [Ollama](https://ollama.com) for local LLM serving
- [Kaggle](https://kaggle.com) for free GPU resources
---

**Happy caching!** 🚀 Star the repo if you find it useful!
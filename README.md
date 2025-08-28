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

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd LLMCache_Proj

# Start with local Ollama
docker-compose up

# In another terminal, pull a model
docker-compose exec ollama ollama pull gemma3:4b

# Run experiments
docker-compose up llmcache-bench
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start local Ollama
ollama serve

# Pull a model
ollama pull gemma3:4b

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

# Fast test run
python scripts/run.py --config test_experiment_fast.yaml
```

### Docker with Custom Config

```bash
# Mount custom config
docker run -v $(pwd)/my_config.yaml:/app/experiments/my_config.yaml \
           -v $(pwd)/results:/app/results \
           llmcache-bench python scripts/run.py --config experiments/my_config.yaml

# Use remote GPU Ollama
docker run -e OLLAMA_BASE_URL=https://your-ngrok-url.com \
           -v $(pwd)/results:/app/results \
           llmcache-bench
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

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/LLMCache_Proj/issues)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/your-username/LLMCache_Proj/discussions)
- 📖 **Documentation**: See `CLAUDE.md` for detailed technical documentation

---

**Happy caching!** 🚀 Star the repo if you find it useful!
# GPTCache + Ollama on Platinum Benchmarks – Project Documentation

## 1. Introduction
This project evaluates the performance impact of using **GPTCache** with a non-OpenAI model (Ollama’s Meta-Llama-3-8B-Instruct) on the **Platinum Benchmarks** dataset, as described in the MadryLab paper ["Do Large Language Model Benchmarks Test Reliability?"](https://huggingface.co/datasets/madrylab/platinum-bench-paper-cache).

The primary objectives are:
- To measure accuracy and performance when using GPTCache to store and reuse responses.
- To identify whether caching improves or degrades performance on Platinum Benchmarks.
- To develop a workflow that runs entirely on local models to avoid API costs.

---

## 2. Project Structure
- **`run_with_gptcache.py`** – Core runner to execute benchmark tests with GPTCache enabled.
- **`GPTCache_basic_test.py`** – Minimal test of GPTCache integration with a non-OpenAI model (Ollama).
- **`run_report.py`** – Custom script to execute benchmarks and store results in Markdown and CSV formats.
- **Dataset:** `madrylab/platinum-bench-paper-cache` (HuggingFace)

---

## 3. Technologies Used
- **GPTCache** – Local caching system for LLM query/response pairs.
- **Ollama** – Local LLM server to run Meta-Llama-3-8B-Instruct (or other supported models) without API cost.
- **Platinum Benchmarks** – Evaluation set for measuring LLM reliability and performance.
- **Python 3** – Execution environment for scripts and benchmarking.
- **Docker (optional)** – Isolates environment and dependencies.

---

## 4. Setup & Installation

### 4.1 Requirements
- Python 3.9+
- Ollama installed locally ([Installation Guide](https://ollama.com/download))
- Git
- HuggingFace Hub access (optional for dataset caching)
- (Optional) Docker for containerized execution

### 4.2 Clone the Project
```bash
git clone <repository_url>
cd GPTCache_PlatinumBenchmarks
```

### 4.3 Install Dependencies
```bash
pip install -r requirements.txt
```

If using Ollama:
```bash
ollama pull llama3.2:1b    # Smaller model for faster tests
# OR
ollama pull llama3:8b      # Larger, more capable model
```

### 4.4 Environment Variables
You can set:
```bash
export OLLAMA_MODEL=llama3.2:1b   # Model choice
export PB_N=3                     # Number of benchmark samples to run
export PB_OUT_DIR=.               # Output directory
```

---

## 5. Running the Project

### 5.1 Start Ollama Server
Make sure Ollama is running only once:
```bash
ollama serve &
```
Verify it’s running:
```bash
ollama ps
```

### 5.2 Execute Benchmarks
Run the report script:
```bash
python run_report.py
```

This will:
- Connect to Ollama.
- Load Platinum Benchmarks.
- Run queries with GPTCache enabled.
- Save results as:
  - `gptcache_platinumbench_report.md` (human-readable summary)
  - `gptcache_platinumbench_runs.csv` (structured results)

---

## 6. Observations & Results

From our initial run with `PB_N=3`:
- The Ollama server loaded **Meta-Llama-3-8B-Instruct** successfully.
- First query took longer due to cold start (model load + first inference ~30s).
- Subsequent queries were faster due to caching.
- HuggingFace Transformers issued a **PyTorch not installed** warning, but since we rely on Ollama, it did not affect execution.

**Impact of GPTCache:**
- GPTCache successfully stored and reused responses for repeated queries.
- With small sample size, performance impact on accuracy was minimal.
- Larger-scale testing is required to determine statistical improvement or degradation.

---

## 7. Conclusions
- GPTCache integration with Ollama on Platinum Benchmarks is functional.
- Caching can reduce response time for repeated queries but may slightly affect accuracy if stale or suboptimal responses are reused.
- Running entirely locally with Ollama eliminates API costs but increases inference latency compared to cloud-hosted LLMs.
- The setup allows further experimentation with:
  - Different caching policies (embedding model, similarity threshold).
  - Different local models (smaller or larger LLaMA variants, Phi models).
  - Varying dataset sample sizes.

---

## 8. Future Work
- Run on **full Platinum Benchmarks dataset** to produce statistically significant results.
- Compare different models for cache embedding computation.
- Analyze trade-offs between cache hit rate and benchmark accuracy.
- Automate result comparison between cached and uncached runs.

---

**Author:** [Your Name]  
**Date:** August 2025

from __future__ import annotations

"""Runner Core - LLMCache-Bench Experiment Runner

Supports dynamic cache strategy loading and execution:
  - load YAML config with cache strategy specification
  - dynamically load cache strategy module by name
  - build LLM (LangChain Ollama adapter) with caching
  - load dataset slice(s) (PlatinumBench subsets)
  - for each row: form question, call LLM, time latency, check correctness
  - aggregate metrics including cache hit rates and write result JSON

Supports any cache strategy in src/cache_strategies/ via dynamic import.
"""

import argparse
import importlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
from tqdm import tqdm

# Local modules
from src.models.ollama import build_llm
from src.bench_datasets.platinum import (
    load as load_platinum,
    pick_question,
    expected_candidates,
)
from src import metrics


# ---------------------------
# Cache Strategy Loading
# ---------------------------

def _load_cache_strategy(cache_mode: str):
    """
    Dynamically import cache strategy module by name.
    
    Args:
        cache_mode: Name of cache strategy (e.g. 'none', 'vanilla_exact', 'extended_loose')
                   Corresponds to src/cache_strategies/{cache_mode}.py
    
    Returns:
        Imported cache strategy module with setup_cache() function
        
    Raises:
        ImportError: If cache strategy module doesn't exist or can't be imported
        AttributeError: If module doesn't have required setup_cache function
    """
    if not cache_mode or cache_mode == "none":
        # Handle none specially since it just disables caching
        cache_mode = "none"
    
    try:
        # Import src.cache_strategies.{cache_mode}
        module_name = f"src.cache_strategies.{cache_mode}"
        cache_module = importlib.import_module(module_name)
        
        # Verify module has required setup_cache function
        if not hasattr(cache_module, 'setup_cache'):
            raise AttributeError(f"Cache strategy '{cache_mode}' missing setup_cache() function")
        
        return cache_module
    
    except ImportError as e:
        available_strategies = _get_available_strategies()
        raise ImportError(
            f"Cache strategy '{cache_mode}' not found. "
            f"Available strategies: {', '.join(available_strategies)}. "
            f"Original error: {e}"
        )


def _get_available_strategies() -> List[str]:
    """Get list of available cache strategy names by scanning cache_strategies directory."""
    strategies_dir = Path("src/cache_strategies")
    if not strategies_dir.exists():
        return []
    
    strategies = []
    for file_path in strategies_dir.glob("*.py"):
        if file_path.name != "__init__.py":
            strategies.append(file_path.stem)  # filename without .py extension
    
    return sorted(strategies)


class _CacheHitTracker:
    """
    Sophisticated cache hit tracking for analyzing cache-induced correctness issues.
    
    Uses multiple signals to accurately detect cache hits:
    1. Response content similarity (primary signal)
    2. Timing patterns (secondary signal) 
    3. Query history tracking for exact and semantic matches
    
    This enables precise "Bad Cache Hit" analysis - cache hits that return wrong answers.
    """
    
    def __init__(self, hit_threshold_sec: float = 0.1):
        self.hit_threshold_sec = hit_threshold_sec
        self.query_history: List[Dict[str, Any]] = []  # Store all queries with responses
        self.hits_detected = 0
        self.bad_hits_detected = 0  # Cache hits that returned wrong answers
        self.total_queries = 0
    
    def _normalize_text(self, s: str) -> str:
        """Normalize text for comparison (matches demo logic)."""
        import re
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[\"'`]", "", s)
        s = re.sub(r"\s([?.!,;:])", r"\1", s)
        return s
    
    def _extract_final_number(self, text: str) -> Optional[float]:
        """Extract final number from text (matches demo logic)."""
        import re
        # Try "Answer: XXX" format first
        m = re.search(r"(?i)answer\s*:\s*(-?\d+(?:\.\d+)?)", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        # Try last number in text
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                return None
        return None
    
    def _same_answer(self, a: str, b: str) -> bool:
        """Check if two responses represent the same answer (matches demo logic)."""
        na, nb = self._extract_final_number(a), self._extract_final_number(b)
        if na is not None and nb is not None:
            return abs(na - nb) <= 1e-6
        return self._normalize_text(a) == self._normalize_text(b)
    
    def record_query(self, latency_sec: float, prompt: str, response: str, is_correct: bool) -> bool:
        """
        Record a query and determine if it was likely a cache hit.
        
        Args:
            latency_sec: Query latency
            prompt: Input prompt 
            response: Model response
            is_correct: Whether the response was correct
            
        Returns:
            True if likely cache hit, False if likely cache miss
        """
        self.total_queries += 1
        
        # Check for cache hits by comparing with previous queries
        is_hit = False
        hit_source = None
        
        for i, prev_query in enumerate(self.query_history):
            prev_prompt = prev_query["prompt"]
            prev_response = prev_query["response"]
            prev_latency = prev_query["latency_sec"]
            
            # Check for exact prompt match (exact cache hit)
            if prompt == prev_prompt and self._same_answer(response, prev_response):
                if latency_sec < max(0.25, prev_latency * 0.7):  # Significantly faster
                    is_hit = True
                    hit_source = f"exact_match_query_{i}"
                    break
            
            # Check for semantic match (semantic cache hit)
            elif prompt != prev_prompt and self._same_answer(response, prev_response):
                if latency_sec < max(0.25, prev_latency * 0.7):  # Significantly faster
                    is_hit = True
                    hit_source = f"semantic_match_query_{i}"
                    break
        
        # Fallback: Use simple timing-based detection if no content-based match found
        if not is_hit and latency_sec < self.hit_threshold_sec:
            is_hit = True
            hit_source = "timing_based"
        
        if is_hit:
            self.hits_detected += 1
            
            # Track "Bad Cache Hits" - cache hits that returned wrong answers
            if not is_correct:
                self.bad_hits_detected += 1
        
        # Store this query for future comparisons
        self.query_history.append({
            "prompt": prompt,
            "response": response,
            "latency_sec": latency_sec,
            "is_correct": is_correct,
            "cache_hit": is_hit,
            "hit_source": hit_source
        })
        
        return is_hit
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate based on content and timing analysis."""
        if self.total_queries == 0:
            return 0.0
        return self.hits_detected / self.total_queries
    
    def get_bad_hit_rate(self) -> float:
        """Get bad cache hit rate (cache hits that returned wrong answers)."""
        if self.hits_detected == 0:
            return 0.0
        return self.bad_hits_detected / self.hits_detected
    
    def get_cache_accuracy(self) -> float:
        """Get accuracy of cache hits (correct cache hits / total cache hits)."""
        if self.hits_detected == 0:
            return 0.0
        correct_hits = self.hits_detected - self.bad_hits_detected
        return correct_hits / self.hits_detected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_misses = self.total_queries - self.hits_detected
        good_cache_hits = self.hits_detected - self.bad_hits_detected
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.hits_detected,
            "cache_misses": cache_misses,
            "bad_cache_hits": self.bad_hits_detected,
            "good_cache_hits": good_cache_hits,
            "hit_rate": self.get_hit_rate(),
            "bad_hit_rate": self.get_bad_hit_rate(),
            "cache_accuracy": self.get_cache_accuracy(),
            "queries_with_hits": len([q for q in self.query_history if q["cache_hit"]])
        }


# ---------------------------
# Public API
# ---------------------------

def main(config_path: str) -> None:
    """Load YAML config, run one experiment, and write JSON results.

    Args:
        config_path: Path to experiments/experiment.yaml
    """
    cfg = _read_yaml(config_path)

    result = run_once(cfg)

    out_dir = Path(cfg.get("output", {}).get("dir", "results/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output file name from pattern
    run_id = cfg.get("run", {}).get("id", f"run-{int(time.time())}")
    pattern = cfg.get("output", {}).get("filename_pattern", "{run_id}.json")
    out_path = out_dir / pattern.format(run_id=run_id)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"✅ Wrote results to {out_path}")


def run_once(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one experiment according to the given config.

    Expects keys (see guide): run, model, dataset, cache, output.
    Supports all cache strategies via dynamic loading.

    Returns:
        Dict matching the example result JSON structure in the guide.
    """
    t_start = time.perf_counter()

    # --- Cache Strategy ---
    cache_cfg = cfg.get("cache", {})
    cache_mode = cache_cfg.get("mode", "none")
    
    print(f"[runner] Loading cache strategy: {cache_mode}")
    try:
        cache_module = _load_cache_strategy(cache_mode)
        # Setup cache before building LLM so LangChain caching is configured
        cache_module.setup_cache(cache_cfg, cfg.get("model", {}))
        print(f"[runner] Cache strategy '{cache_mode}' configured successfully")
    except Exception as e:
        print(f"[runner] Error loading cache strategy '{cache_mode}': {e}")
        raise

    # Initialize cache hit tracker for metrics
    hit_tracker = _CacheHitTracker(hit_threshold_sec=0.1)

    # --- Model ---
    model_cfg = cfg.get("model", {})
    provider = model_cfg.get("provider", "ollama")
    if provider != "ollama":
        raise ValueError("Runner supports provider=ollama only.")

    name = model_cfg.get("name")
    base_url = model_cfg.get("base_url", "http://localhost:11434")
    params = model_cfg.get("params", {})

    llm = build_llm(model=name, base_url=base_url, params=params)
    print(f"[runner] LLM configured: {name} @ {base_url}")

    # --- Dataset ---
    ds_cfg = cfg.get("dataset", {})
    ds_name = ds_cfg.get("name", "platinum-bench")

    # Support either a single `subset` *or* a list under `subsets`.
    subsets = ds_cfg.get("subsets")
    if not subsets:
        subsets = [ds_cfg.get("subset", "gsm8k")]

    split = ds_cfg.get("split", "test")
    slice_cfg = ds_cfg.get("slice", {})
    start = int(slice_cfg.get("start", 0))
    limit = int(slice_cfg.get("limit", 10))

    if ds_name != "platinum-bench":
        raise ValueError("Step 6 supports dataset.name=platinum-bench only.")

    # Load all requested subsets
    loaded: List[Tuple[str, List[Dict[str, Any]]]] = []
    total = 0
    for s in subsets:
        rs = load_platinum(subset=s, split=split, start=start, limit=limit)
        loaded.append((s, rs))
        total += len(rs)

    if total == 0:
        # Still produce a valid, mostly-empty result
        return _empty_result(cfg)

    print(f"[runner] Processing {total} questions across {len(loaded)} subset(s)")

    # --- Main loop ---
    item_logs: List[Dict[str, Any]] = []
    latencies: List[float] = []
    num_correct = 0
    questions_processed = 0

    for subset_name, rows in loaded:
        # Create progress bar for this subset
        desc = f"Processing {subset_name}"
        with tqdm(total=len(rows), desc=desc, unit="questions", ncols=80) as pbar:
            for idx, row in enumerate(rows):
                q = pick_question(row)
                expected = expected_candidates(row, subset_name)

                t0 = time.perf_counter()
                out = llm.invoke(q)
                t1 = time.perf_counter()

                latency = t1 - t0
                latencies.append(latency)

                is_corr = metrics.correctness(expected, out)
                if is_corr:
                    num_correct += 1

                # Detect cache hit using timing heuristics and correctness
                cache_hit = hit_tracker.record_query(latency, q, out, is_correct=is_corr)

                item_logs.append(
                    {
                        "subset": subset_name,
                        "row_index": idx + start,
                        "question": q,
                        "expected": expected,
                        "model_output": out,
                        "latency_sec": round(latency, 6),
                        "cache_hit": cache_hit,
                        "correct": bool(is_corr),
                    }
                )
                
                # Update progress bar with current metrics
                questions_processed += 1
                hit_rate = hit_tracker.get_hit_rate()
                correct_rate = num_correct / questions_processed
                pbar.set_postfix({
                    'correct': f'{correct_rate:.1%}',
                    'cache_hit': f'{hit_rate:.1%}',
                    'latency': f'{latency:.2f}s'
                })
                pbar.update(1)

    elapsed = time.perf_counter() - t_start

    # --- Aggregates ---
    lat_stats = metrics.latency_stats(latencies)
    hitrate = hit_tracker.get_hit_rate()
    bad_hit_rate = hit_tracker.get_bad_hit_rate()
    cache_accuracy = hit_tracker.get_cache_accuracy()
    qps = metrics.throughput(total, elapsed)
    acc = num_correct / total if total else 0.0
    
    print(f"[runner] Completed {total} queries in {elapsed:.2f}s")
    print(f"[runner] Cache hit rate: {hitrate:.2%} ({hit_tracker.hits_detected}/{total})")
    print(f"[runner] Bad cache hit rate: {bad_hit_rate:.2%} ({hit_tracker.bad_hits_detected}/{hit_tracker.hits_detected if hit_tracker.hits_detected > 0 else 1})")
    print(f"[runner] Cache accuracy: {cache_accuracy:.2%}")
    print(f"[runner] Correctness: {acc:.2%} ({num_correct}/{total})")

    # --- Result object ---
    dataset_block = {
        "name": ds_name,
        "split": split,
        "slice": {"start": start, "limit": limit},
    }
    if len(subsets) == 1:
        dataset_block["subset"] = subsets[0]
    else:
        dataset_block["subsets"] = subsets

    result: Dict[str, Any] = {
        "run_id": cfg.get("run", {}).get("id", f"run-{int(time.time())}"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "provider": provider,
            "name": name,
            "base_url": base_url,
            "params": params,
        },
        "dataset": dataset_block,
        "cache": {
            "mode": cache_mode,
            "similarity_threshold": cache_cfg.get("similarity_threshold"),
            "looseness_preset": cache_cfg.get("looseness_preset"),
            "vstore": cache_cfg.get("vstore"),
            "capacity": cache_cfg.get("capacity"),
            "eviction": cache_cfg.get("eviction"),
        },
        "metrics": {
            "latency_mean_sec": round(lat_stats.get("mean", 0.0), 6),
            "latency_p95_sec": round(lat_stats.get("p95", 0.0), 6),
            "latency_p99_sec": round(lat_stats.get("p99", 0.0), 6),
            "cache_hit_rate": hitrate,
            "bad_cache_hit_rate": bad_hit_rate,
            "cache_accuracy": cache_accuracy,
            "cache_statistics": hit_tracker.get_stats(),
            "throughput_qps": round(qps, 6),
            "correctness": round(acc, 6),
        },
        "items": item_logs,
    }

    return result


# ---------------------------
# Helpers
# ---------------------------

def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _empty_result(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a valid result dict when dataset slice is empty."""
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    cache_cfg = cfg.get("cache", {})

    subsets = ds_cfg.get("subsets")
    if not subsets:
        subsets = [ds_cfg.get("subset", "gsm8k")]

    dataset_block = {
        "name": ds_cfg.get("name", "platinum-bench"),
        "split": ds_cfg.get("split", "test"),
        "slice": {
            "start": int(ds_cfg.get("slice", {}).get("start", 0)),
            "limit": int(ds_cfg.get("slice", {}).get("limit", 0)),
        },
    }
    if len(subsets) == 1:
        dataset_block["subset"] = subsets[0]
    else:
        dataset_block["subsets"] = subsets

    return {
        "run_id": cfg.get("run", {}).get("id", f"run-{int(time.time())}"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "provider": model_cfg.get("provider", "ollama"),
            "name": model_cfg.get("name"),
            "base_url": model_cfg.get("base_url", "http://localhost:11434"),
            "params": model_cfg.get("params", {}),
        },
        "dataset": dataset_block,
        "cache": {
            "mode": cache_cfg.get("mode", "none"),
            "similarity_threshold": cache_cfg.get("similarity_threshold"),
            "looseness_preset": cache_cfg.get("looseness_preset"),
            "vstore": cache_cfg.get("vstore"),
            "capacity": cache_cfg.get("capacity"),
            "eviction": cache_cfg.get("eviction"),
        },
        "metrics": {
            "latency_mean_sec": 0.0,
            "latency_p95_sec": 0.0,
            "latency_p99_sec": 0.0,
            "cache_hit_rate": 0.0,
            "throughput_qps": 0.0,
            "correctness": 0.0,
        },
        "items": [],
    }


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one LLMCache-Bench experiment with dynamic cache strategy loading")
    parser.add_argument(
        "config",
        nargs="?",
        default="experiments/experiment.yaml",
        help="Path to the experiment YAML file",
    )
    args = parser.parse_args()
    main(args.config)

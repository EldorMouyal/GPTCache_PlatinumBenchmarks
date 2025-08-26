from __future__ import annotations

"""Runner Core (Step 6)

Mode supported in this step: **none** (no caching).
Keeps the implementation tiny and mirrors the demos' control flow:
  - load YAML config
  - build LLM (LangChain Ollama adapter)
  - load dataset slice(s) (PlatinumBench subsets)
  - for each row: form question, call LLM, time latency, check correctness
  - aggregate metrics and write result JSON

This file deliberately avoids any heavy logging or complex abstractions.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Local modules
from src.models.ollama import build_llm  # Step 4
from src.bench_datasets.platinum import (
    load as load_platinum,
    pick_question,
    expected_candidates,
)  # Step 5
from src import metrics  # Step 2


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
    Only cache.mode == "none" is handled in this step.

    Returns:
        Dict matching the example result JSON structure in the guide.
    """
    t_start = time.perf_counter()

    # --- Model ---
    model_cfg = cfg.get("model", {})
    provider = model_cfg.get("provider", "ollama")
    if provider != "ollama":
        raise ValueError("Step 6 supports provider=ollama only.")

    name = model_cfg.get("name")
    base_url = model_cfg.get("base_url", "http://localhost:11434")
    params = model_cfg.get("params", {})

    llm = build_llm(model=name, base_url=base_url, params=params)

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

    # --- Cache (mode=none only) ---
    cache_cfg = cfg.get("cache", {})
    cache_mode = cache_cfg.get("mode", "none")
    if cache_mode != "none":
        # In this step we intentionally do *not* initialize GPTCache.
        print("[runner] Warning: only cache.mode=none is implemented in Step 6. Proceeding without cache.")

    # --- Main loop ---
    item_logs: List[Dict[str, Any]] = []
    latencies: List[float] = []
    num_correct = 0

    for subset_name, rows in loaded:
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

            item_logs.append(
                {
                    "subset": subset_name,
                    "row_index": idx + start,
                    "question": q,
                    "expected": expected,
                    "model_output": out,
                    "latency_sec": round(latency, 6),
                    "cache_hit": False,  # no cache in this step
                    "correct": bool(is_corr),
                }
            )

    elapsed = time.perf_counter() - t_start

    # --- Aggregates ---
    lat_stats = metrics.latency_stats(latencies)
    hitrate = metrics.hit_rate(0, total)  # no cache => 0 hits
    qps = metrics.throughput(total, elapsed)
    acc = num_correct / total if total else 0.0

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
            # Echo back config; emphasize that cache is disabled in this step.
            "mode": "none",
            "similarity_threshold": cache_cfg.get("similarity_threshold"),
            "vstore": cache_cfg.get("vstore"),
            "capacity": cache_cfg.get("capacity"),
            "eviction": cache_cfg.get("eviction"),
        },
        "metrics": {
            "latency_mean_sec": round(lat_stats.get("mean", 0.0), 6),
            "latency_p95_sec": round(lat_stats.get("p95", 0.0), 6),
            "latency_p99_sec": round(lat_stats.get("p99", 0.0), 6),
            "cache_hit_rate": hitrate,
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
            "mode": "none",
            "similarity_threshold": cache_cfg.get("similarity_threshold"),
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
    parser = argparse.ArgumentParser(description="Run one LLMCache-Bench experiment (mode=none)")
    parser.add_argument(
        "config",
        nargs="?",
        default="experiments/experiment.yaml",
        help="Path to the experiment YAML file",
    )
    args = parser.parse_args()
    main(args.config)

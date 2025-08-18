#!/usr/bin/env python3
"""
runner.py — Step 2 (Runner Core) for LLMCache-Bench

Scope (Step 2 only):
- Load experiments/experiment.yaml
- Validate schema/result.schema.json (structure check)
- Simulate a run over a dummy dataset slice (no real model/dataset/cache)
- Measure per-item latency, produce fake response
- Aggregate minimal metrics: mean/p95/p99 latency (ms), exact-match accuracy (0.0)
- Populate cache stats as zeros under cache.stats
- Emit results JSON to output.dir/{run_id}.json and validate again

Deps: pyyaml, jsonschema (standard library otherwise)
"""

import argparse
import json
import math
import os
import statistics
import time
import uuid
from datetime import datetime, timezone

import yaml
from jsonschema import validate


# ----------------------------- utils -----------------------------

def iso_utc_now() -> str:
    # RFC3339/ISO8601 with 'Z'
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_model_call(prompt: str):
    """Fake model call with tiny non-zero latency and dummy response."""
    start = time.perf_counter()
    time.sleep(0.01)  # ensure latency > 0
    response = f"fake-response-for:{prompt}"
    latency_ms = (time.perf_counter() - start) * 1000.0
    return response, latency_ms


def percentile(values, p):
    """Nearest-rank percentile (safe for small n)."""
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, math.ceil((p / 100.0) * len(s)))
    return float(s[rank - 1])


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="LLMCache-Bench Runner Core (Step 2)")
    parser.add_argument("--config", default="experiments/experiment.yaml",
                        help="Path to experiment YAML (default: experiments/experiment.yaml)")
    parser.add_argument("--schema", default="schema/result.schema.json",
                        help="Path to result JSON schema (default: schema/result.schema.json)")
    args = parser.parse_args()

    # Load config & schema
    cfg = load_yaml(args.config)
    schema = load_json(args.schema)

    run_cfg = cfg.get("run", {})
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    cache_cfg = cfg.get("cache", {})
    output_cfg = cfg.get("output", {})

    # Required-by-schema fields
    run_id = run_cfg.get("id", f"run-{uuid.uuid4().hex[:8]}")
    seed = int(run_cfg.get("seed", 0))
    notes = run_cfg.get("notes")
    slice_cfg = dataset_cfg.get("slice", {}) or {}
    start_idx = int(slice_cfg.get("start", 0))
    limit = int(slice_cfg.get("limit", 1))

    started_at = iso_utc_now()

    items = []
    latencies = []

    # Simulate a slice of dummy prompts, sized by dataset.slice.limit
    for i in range(start_idx, start_idx + limit):
        prompt = f"dummy-prompt-{i}"
        response, latency_ms = simulate_model_call(prompt)
        items.append({
            "id": str(uuid.uuid4()),
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "cached": False
            # "gold" optional; omitted for Step 2
        })
        latencies.append(latency_ms)

    # Minimal metrics (ms)
    latency_mean_ms = float(statistics.mean(latencies)) if latencies else 0.0
    latency_p95_ms = float(percentile(latencies, 95))
    latency_p99_ms = float(percentile(latencies, 99))
    accuracy_exact = 0.0  # trivial placeholder

    completed_at = iso_utc_now()

    # Build result strictly per schema
    run_obj = {
        "id": run_id,
        "seed": seed,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": "success",
    }
    if notes is not None:
        run_obj["notes"] = notes  # allowed by schema

    result = {
        "run": run_obj,
        "model": {
            "provider": model_cfg.get("provider", "dummy"),
            "name": model_cfg.get("name", "dummy-model"),
            # schema requires params object; echo or provide empty object
            "params": model_cfg.get("params", {}),
        },
        "dataset": {
            "name": dataset_cfg.get("name", "dummy-dataset"),
            "split": dataset_cfg.get("split", "dev"),
            "slice": {
                "start": start_idx,
                "limit": limit,
            },
            "count_processed": len(items),
        },
        "cache": {
            "mode": cache_cfg.get("mode", "none"),
            "stats": {
                "hits": 0,
                "misses": 0,
            },
        },
        "metrics": {
            "latency_mean_ms": latency_mean_ms,
            "latency_p95_ms": latency_p95_ms,
            "latency_p99_ms": latency_p99_ms,
            "accuracy_exact": accuracy_exact,
        },
        "items": items,
        "output": {
            "dir": output_cfg.get("dir", "results/raw"),
            # schema requires 'file' (not filename_pattern)
            "file": output_cfg.get("filename_pattern", "{run_id}.json").format(run_id=run_id),
        },
    }

    # Validate BEFORE writing
    try:
        validate(instance=result, schema=schema)
    except Exception as e:
        print("❌ Result does not match schema (pre-write):", e)
        return

    # Prepare output path and write file
    out_dir = result["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, result["output"]["file"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Validate AFTER writing
    try:
        validate(instance=result, schema=schema)
    except Exception as e:
        print("❌ Result failed schema validation after write:", e)
        return

    print(f"✅ Run completed. Output written to {out_path}")
    print("   Validated against schema and OK.")


if __name__ == "__main__":
    main()

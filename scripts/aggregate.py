#!/usr/bin/env python3
"""
scripts/aggregate.py — Cache Reliability Analysis Aggregator

Scans results/raw/*.json and produces results/tables/summary.csv optimized for 
analyzing how caching affects model reliability and correctness.

Focus: Cache impact on model reliability with metrics for plotting comparative analysis.
Format: Current runner.py JSON format only (fails fast on schema mismatches).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd  # type: ignore
except ImportError:
    print(
        "❌ pandas is required for aggregation. Install it with:\n"
        "    pip install pandas\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate LLMCache-Bench results for cache reliability analysis"
    )
    parser.add_argument(
        "--raw-dir", 
        default="results/raw", 
        help="Directory containing raw result JSON files"
    )
    parser.add_argument(
        "--output", 
        default="results/tables/summary.csv",
        help="Output CSV file path"
    )
    return parser.parse_args()


def _validate_result_schema(data: Dict[str, Any], filepath: str) -> None:
    """
    Validate that JSON has expected schema from current runner.py format.
    Fails fast with clear error messages for missing required fields.
    """
    required_top_level = ["run_id", "timestamp", "model", "dataset", "cache", "metrics", "items"]
    for field in required_top_level:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {filepath}")
    
    # Validate nested required fields
    required_model = ["provider", "name", "params"]
    for field in required_model:
        if field not in data["model"]:
            raise ValueError(f"Missing model.{field} in {filepath}")
    
    required_metrics = ["cache_hit_rate", "bad_cache_hit_rate", "cache_accuracy", 
                       "cache_statistics", "correctness", "latency_mean_sec"]
    for field in required_metrics:
        if field not in data["metrics"]:
            raise ValueError(f"Missing metrics.{field} in {filepath}")
    
    required_cache_stats = ["total_queries", "cache_hits", "cache_misses", 
                           "bad_cache_hits", "good_cache_hits"]
    cache_stats = data["metrics"].get("cache_statistics", {})
    for field in required_cache_stats:
        if field not in cache_stats:
            raise ValueError(f"Missing metrics.cache_statistics.{field} in {filepath}")


def _extract_dataset_info(dataset: Dict[str, Any]) -> tuple[str, str, int]:
    """Extract dataset name, subset info, and slice limit."""
    name = dataset.get("name", "unknown")
    
    # Handle both single subset and multiple subsets
    if "subsets" in dataset:
        subsets = dataset["subsets"]
        if isinstance(subsets, list):
            subset_str = ",".join(subsets)
        else:
            subset_str = str(subsets)
    else:
        subset_str = dataset.get("subset", "unknown")
    
    slice_limit = dataset.get("slice", {}).get("limit", 0)
    return name, subset_str, int(slice_limit)


def _calculate_reliability_metrics(metrics: Dict[str, Any], cache_stats: Dict[str, Any]) -> Dict[str, float]:
    """Calculate derived reliability analysis metrics."""
    total_queries = cache_stats["total_queries"]
    cache_hits = cache_stats["cache_hits"]
    bad_cache_hits = cache_stats["bad_cache_hits"]
    hit_rate = metrics["cache_hit_rate"]
    cache_accuracy = metrics["cache_accuracy"]
    correctness = metrics["correctness"]
    
    # Reliability degradation: Direct accuracy loss from bad cache hits
    reliability_degradation = bad_cache_hits / total_queries if total_queries > 0 else 0.0
    
    # Cache effectiveness: Quality-weighted hit rate
    cache_effectiveness = hit_rate * cache_accuracy
    
    # What correctness would be without bad cache hits
    # If we had perfect cache accuracy, bad hits would have been correct
    correctness_without_bad_hits = (correctness * total_queries + bad_cache_hits) / total_queries if total_queries > 0 else correctness
    
    # Cache impact on correctness
    cache_impact_on_correctness = correctness_without_bad_hits - correctness
    
    return {
        "reliability_degradation": reliability_degradation,
        "cache_effectiveness": cache_effectiveness,
        "correctness_without_bad_hits": correctness_without_bad_hits,
        "cache_impact_on_correctness": cache_impact_on_correctness,
    }


def _process_result_file(filepath: str) -> Dict[str, Any] | None:
    """Process a single result JSON file into a row dict."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate schema
        _validate_result_schema(data, filepath)
        
        # Extract basic info
        run_id = data["run_id"]
        timestamp = data["timestamp"]
        
        # Model info
        model = data["model"]
        model_name = model["name"]
        model_provider = model["provider"]
        params = model["params"]
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")
        top_p = params.get("top_p")
        
        # Dataset info
        dataset_name, dataset_subsets, slice_limit = _extract_dataset_info(data["dataset"])
        
        # Cache config
        cache = data["cache"]
        cache_mode = cache["mode"]
        similarity_threshold = cache.get("similarity_threshold")
        looseness_preset = cache.get("looseness_preset")
        
        # Core metrics
        metrics = data["metrics"]
        cache_stats = metrics["cache_statistics"]
        
        # Calculate derived reliability metrics
        reliability_metrics = _calculate_reliability_metrics(metrics, cache_stats)
        
        # Build row
        row = {
            # Experiment identification
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "model_provider": model_provider,
            "cache_mode": cache_mode,
            "dataset_name": dataset_name,
            "dataset_subsets": dataset_subsets,
            "slice_limit": slice_limit,
            
            # Model parameters
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            
            # Cache configuration
            "similarity_threshold": similarity_threshold,
            "looseness_preset": looseness_preset,
            
            # Core reliability metrics
            "correctness": metrics["correctness"],
            "cache_hit_rate": metrics["cache_hit_rate"],
            "bad_cache_hit_rate": metrics["bad_cache_hit_rate"],
            "cache_accuracy": metrics["cache_accuracy"],
            
            # Raw counts for analysis
            "total_queries": cache_stats["total_queries"],
            "cache_hits": cache_stats["cache_hits"],
            "cache_misses": cache_stats["cache_misses"],
            "bad_cache_hits": cache_stats["bad_cache_hits"],
            "good_cache_hits": cache_stats["good_cache_hits"],
            
            # Performance metrics
            "latency_mean_sec": metrics["latency_mean_sec"],
            "latency_p95_sec": metrics.get("latency_p95_sec"),
            "throughput_qps": metrics.get("throughput_qps"),
            
            # Calculated reliability analysis
            "reliability_degradation": reliability_metrics["reliability_degradation"],
            "cache_effectiveness": reliability_metrics["cache_effectiveness"],
            "correctness_without_bad_hits": reliability_metrics["correctness_without_bad_hits"],
            "cache_impact_on_correctness": reliability_metrics["cache_impact_on_correctness"],
            
            # File reference
            "result_file": os.path.relpath(filepath),
        }
        
        return row
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}", file=sys.stderr)
        return None


def main() -> None:
    args = _parse_args()
    
    # Find all JSON files
    json_pattern = os.path.join(args.raw_dir, "*.json")
    json_files = sorted(glob.glob(json_pattern))
    
    if not json_files:
        print(f"❌ No JSON files found in {args.raw_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(json_files)} result files from {args.raw_dir}")
    
    # Process all files
    rows = []
    for filepath in json_files:
        row = _process_result_file(filepath)
        if row:
            rows.append(row)
    
    if not rows:
        print("❌ No valid result files found", file=sys.stderr)
        sys.exit(1)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(args.output, index=False)
    
    print(f"✅ Aggregated {len(df)} experiments to {args.output}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Cache modes: {sorted(df['cache_mode'].unique())}")
    print(f"   Models: {sorted(df['model_name'].unique())}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Cache Optimization Script

Automated parameter sweep for enhanced_semantic cache strategy to find optimal
embedding model and similarity threshold combinations. Tests multiple configurations,
clears cache between runs, and provides comprehensive analysis of results.
"""

import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import yaml
from datetime import datetime


# Configuration
EMBEDDING_MODELS = ["onnx", "bge_small", "e5_base", "mpnet"]
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
BASE_CONFIG_TEMPLATE = "experiments/experiment.yaml"
OUTPUT_DIR = "results/optimization"
TEMP_CONFIG_DIR = "temp_configs"


def ensure_directories():
    """Create necessary directories for optimization."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_CONFIG_DIR).mkdir(parents=True, exist_ok=True)


def clear_all_caches():
    """Clear all cache directories before running experiments."""
    print("[optimize] Clearing all cache directories...")
    cache_dirs = [
        "cache",
        "cache/enhanced_semantic_onnx_cache",
        "cache/enhanced_semantic_bge_small_cache", 
        "cache/enhanced_semantic_e5_base_cache",
        "cache/enhanced_semantic_mpnet_cache"
    ]
    
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
            print(f"[optimize] Removed {cache_dir}")
    
    # Also remove any .db and .index files
    for pattern in ["*.db", "*.index"]:
        for file_path in Path(".").glob(pattern):
            if "cache" in str(file_path):
                file_path.unlink()
                print(f"[optimize] Removed {file_path}")


def generate_config(embedding_model: str, threshold: float, base_config_path: str) -> str:
    """
    Generate experiment configuration file for specific parameters.
    
    Args:
        embedding_model: Embedding model key (onnx, bge_small, e5_base, mpnet)
        threshold: Similarity threshold (0.8-0.95)
        base_config_path: Path to base configuration template
        
    Returns:
        Path to generated configuration file
    """
    # Load base configuration as YAML
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Generate unique run ID
    run_id = f"enhanced_{embedding_model}_{threshold:.2f}"
    
    # Update configuration values
    # Update run ID (flexible - works with any existing ID)
    if 'run' not in config:
        config['run'] = {}
    config['run']['id'] = run_id
    
    # Update cache strategy to enhanced_semantic
    if 'cache' not in config:
        config['cache'] = {}
    config['cache']['mode'] = 'enhanced_semantic'
    config['cache']['similarity_threshold'] = threshold
    config['cache']['embedding_model'] = embedding_model
    config['cache']['enable_quality_validation'] = True
    
    # Preserve other cache settings but ensure we have defaults
    if 'vstore' not in config['cache']:
        config['cache']['vstore'] = 'faiss'
    if 'capacity' not in config['cache']:
        config['cache']['capacity'] = 5000
    if 'eviction' not in config['cache']:
        config['cache']['eviction'] = 'LRU'
    
    # Write configuration file
    config_filename = f"{TEMP_CONFIG_DIR}/config_{run_id}.yaml"
    with open(config_filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    return config_filename


def run_experiment(config_path: str) -> Tuple[bool, str]:
    """
    Run a single experiment with given configuration.
    
    Args:
        config_path: Path to experiment configuration file
        
    Returns:
        (success, result_file_path) tuple
    """
    print(f"[optimize] Running experiment with config: {config_path}")
    
    try:
        # Run experiment
        result = subprocess.run(
            [sys.executable, "scripts/run.py", "--config", config_path],
            capture_output=True, # set to False if you want to see status and more information.
            text=True
        )
        
        if result.returncode != 0:
            print(f"[optimize] Experiment failed with return code {result.returncode}")
            print(f"[optimize] Error output: {result.stderr}")
            return False, ""
        
        # Find result file - extract run_id from config path
        run_id = Path(config_path).stem.replace("config_", "")
        result_file = f"results/raw/{run_id}.json"
        
        if Path(result_file).exists():
            return True, result_file
        else:
            print(f"[optimize] Result file not found: {result_file}")
            return False, ""
            
    except Exception as e:
        print(f"[optimize] Experiment failed with exception: {e}")
        return False, ""


def run_aggregation_and_plotting() -> Tuple[bool, str]:
    """
    Run aggregate.py and plot.py scripts to generate analysis.
    
    Returns:
        (success, csv_path) tuple
    """
    print("[optimize] Running aggregation analysis...")
    
    # Ensure output directories exist
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    csv_path = "results/tables/summary.csv"
    
    try:
        # Run aggregation
        result = subprocess.run([
            sys.executable, "scripts/aggregate.py",
            "--raw-dir", "results/raw",
            "--output", csv_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[optimize] Aggregation failed: {result.stderr}")
            return False, ""
        
        print(f"[optimize] ✓ Aggregation complete: {csv_path}")
        
        # Run plotting
        result = subprocess.run([
            sys.executable, "scripts/plot.py",
            "--csv", csv_path,
            "--out-dir", "results/figures"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[optimize] Plotting failed: {result.stderr}")
            return False, ""
        
        print(f"[optimize] ✓ Plots generated: results/figures/")
        return True, csv_path
        
    except Exception as e:
        print(f"[optimize] Analysis failed: {e}")
        return False, ""


def run_optimization_sweep():
    """Run complete optimization sweep across all parameter combinations."""
    print("[optimize] Starting cache optimization sweep...")
    print(f"[optimize] Testing {len(EMBEDDING_MODELS)} models × {len(THRESHOLDS)} thresholds = {len(EMBEDDING_MODELS) * len(THRESHOLDS)} configurations")
    
    ensure_directories()
    successful_runs = 0
    
    total_configs = len(EMBEDDING_MODELS) * len(THRESHOLDS)
    current_config = 0
    
    for embedding_model in EMBEDDING_MODELS:
        for threshold in THRESHOLDS:
            current_config += 1
            print(f"\n[optimize] === Configuration {current_config}/{total_configs}: {embedding_model} @ {threshold} ===")
            
            # Clear caches before each run
            clear_all_caches()
            
            # Generate configuration
            config_path = generate_config(embedding_model, threshold, BASE_CONFIG_TEMPLATE)
            
            # Run experiment
            success, result_file = run_experiment(config_path)
            
            if success:
                successful_runs += 1
                print(f"[optimize] ✓ Success: Results saved to {result_file}")
            else:
                print(f"[optimize] ✗ Failed: {embedding_model} @ {threshold}")
    
    print(f"\n[optimize] Completed {successful_runs}/{total_configs} experiments")
    return successful_runs


def generate_optimization_summary(csv_path: str):
    """Generate optimization-specific summary from aggregated data."""
    try:
        df = pd.read_csv(csv_path)
        
        # Filter to only optimization runs (enhanced_* pattern)
        optimization_runs = df[df['run_id'].str.startswith('enhanced_')]
        
        if optimization_runs.empty:
            print("[optimize] No optimization runs found in aggregated data")
            return
        
        print("\n" + "="*80)
        print("CACHE OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nProcessed {len(optimization_runs)} optimization configurations")
        print(f"Cache strategies tested: {sorted(optimization_runs['cache_mode'].unique())}")
        
        print("\nTop 5 configurations by cache hit rate:")
        top_hit_rate = optimization_runs.nlargest(5, 'cache_hit_rate')
        for _, row in top_hit_rate.iterrows():
            model_threshold = f"{row.get('similarity_threshold', 'N/A')}"
            print(f"  {row['cache_mode']:<20} @ {model_threshold}: "
                  f"{row['cache_hit_rate']:.1%} hits, {row['bad_cache_hit_rate']:.1%} bad hits, "
                  f"{row['cache_accuracy']:.1%} accuracy")
        
        print("\nTop 5 configurations by cache accuracy:")
        top_accuracy = optimization_runs.nlargest(5, 'cache_accuracy')
        for _, row in top_accuracy.iterrows():
            model_threshold = f"{row.get('similarity_threshold', 'N/A')}"
            print(f"  {row['cache_mode']:<20} @ {model_threshold}: "
                  f"{row['cache_accuracy']:.1%} accuracy, {row['cache_hit_rate']:.1%} hits, "
                  f"{row['bad_cache_hit_rate']:.1%} bad hits")
        
        print("\nLowest bad cache hit rates:")
        low_bad_hits = optimization_runs.nsmallest(5, 'bad_cache_hit_rate')
        for _, row in low_bad_hits.iterrows():
            model_threshold = f"{row.get('similarity_threshold', 'N/A')}"
            print(f"  {row['cache_mode']:<20} @ {model_threshold}: "
                  f"{row['bad_cache_hit_rate']:.1%} bad hits, {row['cache_hit_rate']:.1%} hits, "
                  f"{row['cache_accuracy']:.1%} accuracy")
        
        # Find optimal configuration using cache effectiveness score
        optimization_runs['optimization_score'] = optimization_runs['cache_effectiveness']
        optimal = optimization_runs.loc[optimization_runs['optimization_score'].idxmax()]
        
        print(f"\nRECOMMENDED OPTIMAL CONFIGURATION:")
        print(f"  Run ID: {optimal['run_id']}")
        print(f"  Cache Mode: {optimal['cache_mode']}")
        print(f"  Similarity Threshold: {optimal.get('similarity_threshold', 'N/A')}")
        print(f"  Cache Hit Rate: {optimal['cache_hit_rate']:.1%}")
        print(f"  Bad Cache Hit Rate: {optimal['bad_cache_hit_rate']:.1%}")
        print(f"  Cache Accuracy: {optimal['cache_accuracy']:.1%}")
        print(f"  Overall Correctness: {optimal['correctness']:.1%}")
        print(f"  Cache Effectiveness: {optimal['cache_effectiveness']:.3f}")
        
    except Exception as e:
        print(f"[optimize] Error generating summary: {e}")




def cleanup_temp_configs():
    """Remove temporary configuration files."""
    if Path(TEMP_CONFIG_DIR).exists():
        shutil.rmtree(TEMP_CONFIG_DIR)
        print(f"[optimize] Cleaned up temporary configs")


def main():
    """Main optimization workflow."""
    print("Cache Strategy Optimization Tool")
    print("="*50)
    
    try:
        # Run optimization sweep
        successful_runs = run_optimization_sweep()
        
        if successful_runs > 0:
            print(f"\n[optimize] {successful_runs} experiments completed successfully")
            
            # Run aggregation and plotting using existing scripts
            success, csv_path = run_aggregation_and_plotting()
            
            if success:
                # Generate optimization-specific summary
                generate_optimization_summary(csv_path)
                print(f"\n[optimize] ✅ Analysis complete!")
                print(f"   - Data: {csv_path}")
                print(f"   - Plots: results/figures/")
            else:
                print("[optimize] ❌ Analysis failed - check logs above")
        else:
            print("[optimize] No successful experiments completed!")
            
    except KeyboardInterrupt:
        print("\n[optimize] Optimization interrupted by user")
    except Exception as e:
        print(f"[optimize] Optimization failed with error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_temp_configs()
        print("\n[optimize] Optimization complete!")


if __name__ == "__main__":
    main()
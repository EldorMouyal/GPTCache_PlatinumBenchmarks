#!/usr/bin/env python3
"""
Cache Optimization Script

Automated parameter sweep for enhanced_semantic cache strategy to find optimal
embedding model and similarity threshold combinations. Tests multiple configurations,
clears cache between runs, and provides comprehensive analysis of results.
"""

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    # Read base configuration
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Generate unique run ID
    run_id = f"enhanced_{embedding_model}_{threshold:.2f}"
    
    # Update configuration
    config_content = config_content.replace(
        'id: "vanilla_approx_095"',
        f'id: "{run_id}"'
    )
    config_content = config_content.replace(
        'mode: "vanilla_approx"',
        'mode: "enhanced_semantic"'
    )
    config_content = config_content.replace(
        'similarity_threshold: 0.95',
        f'similarity_threshold: {threshold}'
    )
    
    # Add embedding model configuration
    # Find cache section and add embedding_model parameter
    lines = config_content.split('\n')
    cache_section_found = False
    modified_lines = []
    
    for line in lines:
        modified_lines.append(line)
        if line.strip() == "cache:" or line.strip().startswith("cache:"):
            cache_section_found = True
        elif cache_section_found and line.strip().startswith("similarity_threshold:"):
            # Add embedding_model right after similarity_threshold
            indent = "  "  # Match YAML indentation
            modified_lines.append(f"{indent}embedding_model: \"{embedding_model}\"")
            modified_lines.append(f"{indent}enable_quality_validation: true")
            cache_section_found = False  # Only add once
    
    config_content = '\n'.join(modified_lines)
    
    # Write configuration file
    config_filename = f"{TEMP_CONFIG_DIR}/config_{run_id}.yaml"
    with open(config_filename, 'w') as f:
        f.write(config_content)
    
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
            capture_output=True,
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


def extract_metrics(result_file: str) -> Dict[str, Any]:
    """
    Extract relevant metrics from experiment result file.
    
    Args:
        result_file: Path to experiment result JSON file
        
    Returns:
        Dictionary with extracted metrics
    """
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    cache_stats = metrics.get('cache_statistics', {})
    
    return {
        'cache_hit_rate': metrics.get('cache_hit_rate', 0.0),
        'bad_cache_hit_rate': metrics.get('bad_cache_hit_rate', 0.0),
        'cache_accuracy': metrics.get('cache_accuracy', 0.0),
        'correctness': metrics.get('correctness', 0.0),
        'total_queries': cache_stats.get('total_queries', 0),
        'cache_hits': cache_stats.get('cache_hits', 0),
        'bad_cache_hits': cache_stats.get('bad_cache_hits', 0),
        'good_cache_hits': cache_stats.get('good_cache_hits', 0),
        'latency_mean_sec': metrics.get('latency_mean_sec', 0.0),
        'throughput_qps': metrics.get('throughput_qps', 0.0)
    }


def run_optimization_sweep():
    """Run complete optimization sweep across all parameter combinations."""
    print("[optimize] Starting cache optimization sweep...")
    print(f"[optimize] Testing {len(EMBEDDING_MODELS)} models × {len(THRESHOLDS)} thresholds = {len(EMBEDDING_MODELS) * len(THRESHOLDS)} configurations")
    
    ensure_directories()
    results = []
    
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
                # Extract metrics
                metrics = extract_metrics(result_file)
                metrics.update({
                    'embedding_model': embedding_model,
                    'threshold': threshold,
                    'run_id': f"enhanced_{embedding_model}_{threshold:.2f}"
                })
                results.append(metrics)
                
                print(f"[optimize] ✓ Success: {metrics['cache_hit_rate']:.1%} hit rate, {metrics['bad_cache_hit_rate']:.1%} bad hit rate")
            else:
                print(f"[optimize] ✗ Failed: {embedding_model} @ {threshold}")
    
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze optimization results and generate summary."""
    if not results:
        print("[optimize] No successful results to analyze!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{OUTPUT_DIR}/optimization_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[optimize] Saved results to: {csv_path}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    print("\nTop 5 configurations by cache hit rate:")
    top_hit_rate = df.nlargest(5, 'cache_hit_rate')
    for _, row in top_hit_rate.iterrows():
        print(f"  {row['embedding_model']:<10} @ {row['threshold']:.2f}: "
              f"{row['cache_hit_rate']:.1%} hits, {row['bad_cache_hit_rate']:.1%} bad hits, "
              f"{row['cache_accuracy']:.1%} accuracy")
    
    print("\nTop 5 configurations by cache accuracy:")
    top_accuracy = df.nlargest(5, 'cache_accuracy')
    for _, row in top_accuracy.iterrows():
        print(f"  {row['embedding_model']:<10} @ {row['threshold']:.2f}: "
              f"{row['cache_accuracy']:.1%} accuracy, {row['cache_hit_rate']:.1%} hits, "
              f"{row['bad_cache_hit_rate']:.1%} bad hits")
    
    print("\nLowest bad cache hit rates:")
    low_bad_hits = df.nsmallest(5, 'bad_cache_hit_rate')
    for _, row in low_bad_hits.iterrows():
        print(f"  {row['embedding_model']:<10} @ {row['threshold']:.2f}: "
              f"{row['bad_cache_hit_rate']:.1%} bad hits, {row['cache_hit_rate']:.1%} hits, "
              f"{row['cache_accuracy']:.1%} accuracy")
    
    # Find optimal configuration (balance hit rate and accuracy)
    # Score = cache_hit_rate * (1 - bad_cache_hit_rate)  
    df['optimization_score'] = df['cache_hit_rate'] * (1 - df['bad_cache_hit_rate'])
    optimal = df.loc[df['optimization_score'].idxmax()]
    
    print(f"\nRECOMMENDED OPTIMAL CONFIGURATION:")
    print(f"  Model: {optimal['embedding_model']}")
    print(f"  Threshold: {optimal['threshold']:.2f}")
    print(f"  Cache Hit Rate: {optimal['cache_hit_rate']:.1%}")
    print(f"  Bad Cache Hit Rate: {optimal['bad_cache_hit_rate']:.1%}")
    print(f"  Cache Accuracy: {optimal['cache_accuracy']:.1%}")
    print(f"  Overall Correctness: {optimal['correctness']:.1%}")
    print(f"  Optimization Score: {optimal['optimization_score']:.3f}")
    
    # Generate plots
    generate_plots(df, timestamp)


def generate_plots(df: pd.DataFrame, timestamp: str):
    """Generate visualization plots for optimization results."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cache Strategy Optimization Results', fontsize=16, y=0.98)
    
    # Plot 1: Cache Hit Rate by Model and Threshold
    pivot_hit = df.pivot(index='embedding_model', columns='threshold', values='cache_hit_rate')
    sns.heatmap(pivot_hit, annot=True, fmt='.1%', ax=axes[0,0], cmap='YlOrRd')
    axes[0,0].set_title('Cache Hit Rate')
    axes[0,0].set_xlabel('Threshold')
    axes[0,0].set_ylabel('Embedding Model')
    
    # Plot 2: Bad Cache Hit Rate by Model and Threshold
    pivot_bad = df.pivot(index='embedding_model', columns='threshold', values='bad_cache_hit_rate')
    sns.heatmap(pivot_bad, annot=True, fmt='.1%', ax=axes[0,1], cmap='YlOrRd_r')
    axes[0,1].set_title('Bad Cache Hit Rate (Lower is Better)')
    axes[0,1].set_xlabel('Threshold')
    axes[0,1].set_ylabel('Embedding Model')
    
    # Plot 3: Cache Hit Rate vs Bad Cache Hit Rate Scatter
    colors = {'onnx': 'red', 'bge_small': 'blue', 'e5_base': 'green', 'mpnet': 'orange'}
    for model in df['embedding_model'].unique():
        model_data = df[df['embedding_model'] == model]
        axes[1,0].scatter(model_data['cache_hit_rate'], model_data['bad_cache_hit_rate'], 
                         label=model, alpha=0.7, s=80, c=colors.get(model, 'gray'))
    
    axes[1,0].set_xlabel('Cache Hit Rate')
    axes[1,0].set_ylabel('Bad Cache Hit Rate')
    axes[1,0].set_title('Cache Performance Trade-off')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Optimization Score by Configuration
    df_sorted = df.sort_values('optimization_score', ascending=True)
    y_pos = range(len(df_sorted))
    bars = axes[1,1].barh(y_pos, df_sorted['optimization_score'])
    
    # Color bars by embedding model
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        bars[i].set_color(colors.get(row['embedding_model'], 'gray'))
    
    axes[1,1].set_yticks(y_pos)
    axes[1,1].set_yticklabels([f"{row['embedding_model']}@{row['threshold']:.2f}" 
                               for _, row in df_sorted.iterrows()])
    axes[1,1].set_xlabel('Optimization Score (Hit Rate × Cache Accuracy)')
    axes[1,1].set_title('Overall Performance Ranking')
    axes[1,1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{OUTPUT_DIR}/optimization_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[optimize] Saved plots to: {plot_path}")


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
        results = run_optimization_sweep()
        
        if results:
            # Analyze and visualize results
            analyze_results(results)
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
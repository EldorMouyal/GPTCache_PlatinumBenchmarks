#!/usr/bin/env python3
"""
scripts/plot_embedding_comparison.py — Embedding Model Analysis Visualizations

Reads the aggregated CSV and generates 4 charts focused on embedding model comparison:
1. Correctness by Embedding Model & Threshold - Bar chart grouped by model and threshold
2. Latency by Embedding Model & Threshold - Performance impact comparison  
3. Cache Accuracy by Embedding Model & Threshold - Cache quality comparison
4. Multi-Metric Summary - Combined dashboard showing key metrics across models

Designed for systematic comparison of embedding models (onnx, bge_small, e5_base, mpnet)
with threshold analysis (0.9 vs 0.95) and no-cache baseline reference.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any

# Required dependencies
try:
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    import seaborn as sns  # type: ignore
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"❌ {missing} is required. Install with: pip install {missing}", file=sys.stderr)
    sys.exit(1)


# Required columns for embedding model analysis
REQUIRED_COLS = [
    "cache_mode",
    "embedding_model", 
    "similarity_threshold",
    "correctness", 
    "cache_hit_rate",
    "cache_accuracy",
    "latency_mean_sec"
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embedding model comparison charts"
    )
    parser.add_argument(
        "--csv", 
        default="results/tables/summary.csv", 
        help="Input aggregated CSV file"
    )
    parser.add_argument(
        "--out-dir", 
        default="results/figures", 
        help="Output directory for figures"
    )
    return parser.parse_args()


def _validate_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    if not os.path.isfile(csv_path):
        print(f"❌ CSV file not found: {csv_path}. Run scripts/aggregate.py first.", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("❌ CSV file is empty. Run some experiments first.", file=sys.stderr)
        sys.exit(1)
    
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    # Convert numeric columns
    numeric_cols = ["similarity_threshold", "correctness", "cache_hit_rate", 
                   "cache_accuracy", "latency_mean_sec"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def _ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def _get_cached_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get only cached experiments (exclude 'none' cache mode)."""
    return df[df['cache_mode'] != 'none'].copy()


def _get_nocache_baseline(df: pd.DataFrame) -> float:
    """Get correctness baseline from no-cache experiments."""
    nocache = df[df['cache_mode'] == 'none']
    if nocache.empty:
        return None
    return nocache['correctness'].mean()


def _plot_correctness_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 1: Correctness by Embedding Model & Threshold."""
    cached_df = _get_cached_data(df)
    nocache_baseline = _get_nocache_baseline(df)
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping correctness comparison")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart grouped by embedding model and threshold
    if 'embedding_model' in cached_df.columns and not cached_df['embedding_model'].isna().all():
        # Group by embedding model and threshold
        grouped = cached_df.groupby(['embedding_model', 'similarity_threshold'])['correctness'].mean().reset_index()
        
        # Pivot for grouped bar chart
        pivot_data = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='correctness')
        
        pivot_data.plot(kind='bar', ax=ax1, width=0.8, alpha=0.8)
        ax1.set_title('Correctness by Embedding Model & Threshold')
        ax1.set_xlabel('Embedding Model')
        ax1.set_ylabel('Correctness')
        ax1.legend(title='Similarity Threshold')
        ax1.grid(True, alpha=0.3)
        
        # Add baseline line if available
        if nocache_baseline is not None:
            ax1.axhline(y=nocache_baseline, color='red', linestyle='--', 
                       label=f'No Cache Baseline ({nocache_baseline:.3f})')
            ax1.legend()
        
        ax1.tick_labels = ax1.get_xticklabels()
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Plot 2: Comparison with baseline
    if nocache_baseline is not None:
        model_means = cached_df.groupby('embedding_model')['correctness'].mean()
        
        x_pos = np.arange(len(model_means))
        bars = ax2.bar(x_pos, model_means.values, alpha=0.7)
        ax2.axhline(y=nocache_baseline, color='red', linestyle='--', 
                   label=f'No Cache Baseline ({nocache_baseline:.3f})')
        
        # Color bars based on performance vs baseline
        for bar, value in zip(bars, model_means.values):
            if value < nocache_baseline:
                bar.set_color('lightcoral')
            else:
                bar.set_color('lightgreen')
        
        ax2.set_title('Cached vs No-Cache Correctness')
        ax2.set_xlabel('Embedding Model')
        ax2.set_ylabel('Correctness')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_means.index, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "correctness_by_model_threshold.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_latency_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 2: Latency by Embedding Model & Threshold."""
    cached_df = _get_cached_data(df)
    nocache_baseline = df[df['cache_mode'] == 'none']['latency_mean_sec'].mean() if 'none' in df['cache_mode'].values else None
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping latency comparison")
        return
    
    plt.figure(figsize=(12, 6))
    
    if 'embedding_model' in cached_df.columns and not cached_df['embedding_model'].isna().all():
        # Group by embedding model and threshold
        grouped = cached_df.groupby(['embedding_model', 'similarity_threshold'])['latency_mean_sec'].mean().reset_index()
        
        # Pivot for grouped bar chart
        pivot_data = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='latency_mean_sec')
        
        ax = pivot_data.plot(kind='bar', width=0.8, alpha=0.8, figsize=(12, 6))
        plt.title('Latency by Embedding Model & Threshold')
        plt.xlabel('Embedding Model')
        plt.ylabel('Latency (seconds)')
        plt.legend(title='Similarity Threshold')
        plt.grid(True, alpha=0.3)
        
        # Add baseline line if available
        if nocache_baseline is not None:
            plt.axhline(y=nocache_baseline, color='red', linestyle='--', 
                       label=f'No Cache Baseline ({nocache_baseline:.3f}s)')
            plt.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    output_path = os.path.join(output_dir, "latency_by_model_threshold.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_cache_accuracy_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 3: Cache Accuracy by Embedding Model & Threshold."""
    cached_df = _get_cached_data(df)
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping cache accuracy comparison")
        return
    
    plt.figure(figsize=(12, 6))
    
    if 'embedding_model' in cached_df.columns and not cached_df['embedding_model'].isna().all():
        # Group by embedding model and threshold
        grouped = cached_df.groupby(['embedding_model', 'similarity_threshold'])['cache_accuracy'].mean().reset_index()
        
        # Pivot for grouped bar chart
        pivot_data = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='cache_accuracy')
        
        ax = pivot_data.plot(kind='bar', width=0.8, alpha=0.8, figsize=(12, 6))
        plt.title('Cache Accuracy by Embedding Model & Threshold')
        plt.xlabel('Embedding Model')
        plt.ylabel('Cache Accuracy')
        plt.legend(title='Similarity Threshold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cache_accuracy_by_model_threshold.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_multi_metric_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 4: Multi-Metric Summary Dashboard."""
    cached_df = _get_cached_data(df)
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping multi-metric summary")
        return
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    if 'embedding_model' in cached_df.columns and not cached_df['embedding_model'].isna().all():
        # Group data by embedding model
        model_groups = cached_df.groupby('embedding_model')
        
        # Metric 1: Correctness
        correctness_means = model_groups['correctness'].mean()
        ax1.bar(correctness_means.index, correctness_means.values, alpha=0.7, color='skyblue')
        ax1.set_title('Average Correctness by Model')
        ax1.set_ylabel('Correctness')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Metric 2: Latency  
        latency_means = model_groups['latency_mean_sec'].mean()
        ax2.bar(latency_means.index, latency_means.values, alpha=0.7, color='lightcoral')
        ax2.set_title('Average Latency by Model')
        ax2.set_ylabel('Latency (seconds)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Metric 3: Cache Hit Rate
        hitrate_means = model_groups['cache_hit_rate'].mean()
        ax3.bar(hitrate_means.index, hitrate_means.values, alpha=0.7, color='lightgreen')
        ax3.set_title('Average Cache Hit Rate by Model')
        ax3.set_ylabel('Cache Hit Rate')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Metric 4: Cache Accuracy
        accuracy_means = model_groups['cache_accuracy'].mean()
        ax4.bar(accuracy_means.index, accuracy_means.values, alpha=0.7, color='gold')
        ax4.set_title('Average Cache Accuracy by Model')
        ax4.set_ylabel('Cache Accuracy')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "multi_metric_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main() -> None:
    args = _parse_args()
    
    # Load and validate data
    df = _validate_csv(args.csv)
    _ensure_output_dir(args.out_dir)
    
    print(f"Processing {len(df)} experiments from {args.csv}")
    print(f"Cache strategies: {sorted(df['cache_mode'].unique())}")
    if 'embedding_model' in df.columns:
        embedding_models = df['embedding_model'].dropna().unique()
        if len(embedding_models) > 0:
            print(f"Embedding models: {sorted(embedding_models)}")
    
    # Generate charts
    _plot_correctness_by_model_threshold(df, args.out_dir)
    _plot_latency_by_model_threshold(df, args.out_dir)
    _plot_cache_accuracy_by_model_threshold(df, args.out_dir)
    _plot_multi_metric_summary(df, args.out_dir)
    
    print(f"✅ All embedding comparison charts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
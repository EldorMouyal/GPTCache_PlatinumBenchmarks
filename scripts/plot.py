#!/usr/bin/env python3
"""
scripts/plot.py — Cache Reliability Analysis Visualizations

Reads the aggregated CSV and generates 4 simple charts focused on cache reliability research:
1. Hit Rate vs Correctness - Core trade-off analysis
2. Cache Impact by Strategy - Accuracy loss comparison  
3. Cache Quality Metrics - Reliability metrics by strategy
4. Threshold Sensitivity - How similarity thresholds affect performance

Clean, straightforward visualizations for cache reliability analysis.
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
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"❌ {missing} is required. Install with: pip install {missing}", file=sys.stderr)
    sys.exit(1)


# Required columns for cache reliability analysis
REQUIRED_COLS = [
    "cache_mode",
    "correctness", 
    "cache_hit_rate",
    "bad_cache_hit_rate",
    "cache_accuracy",
    "cache_impact_on_correctness",
    "latency_mean_sec"
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cache reliability analysis charts"
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
    numeric_cols = ["correctness", "cache_hit_rate", "bad_cache_hit_rate", 
                   "cache_accuracy", "cache_impact_on_correctness", "latency_mean_sec"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                print(f"❌ Column {col} has no valid numeric data", file=sys.stderr)
                sys.exit(1)
    
    return df


def _ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def _plot_hit_rate_vs_correctness(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 1: Hit Rate vs Correctness scatter plot."""
    plt.figure(figsize=(10, 6))
    
    # Group by cache_mode for different colors
    cache_modes = df['cache_mode'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cache_modes)))
    
    for i, mode in enumerate(cache_modes):
        mode_data = df[df['cache_mode'] == mode]
        plt.scatter(mode_data['cache_hit_rate'], mode_data['correctness'], 
                   c=[colors[i]], label=mode, alpha=0.7, s=60)
    
    plt.xlabel('Cache Hit Rate')
    plt.ylabel('Model Correctness')
    plt.title('Cache Hit Rate vs Model Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "hit_rate_vs_correctness.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_cache_impact_by_strategy(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 2: Cache Impact by Strategy bar chart."""
    # Aggregate by cache_mode
    grouped = df.groupby('cache_mode')['cache_impact_on_correctness'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped.index, grouped.values)
    
    # Color bars based on impact (red = bad impact, green = good/no impact)
    for bar, value in zip(bars, grouped.values):
        if value > 0:
            bar.set_color('red')
        else:
            bar.set_color('lightblue')
    
    plt.xlabel('Cache Strategy')
    plt.ylabel('Cache Impact on Correctness')
    plt.title('Accuracy Loss by Cache Strategy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, grouped.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cache_impact_by_strategy.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_cache_quality_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 3: Cache Quality Metrics grouped bar chart."""
    # Aggregate by cache_mode
    grouped = df.groupby('cache_mode')[['bad_cache_hit_rate', 'cache_accuracy']].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(grouped.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, grouped['bad_cache_hit_rate'], width, 
                   label='Bad Hit Rate', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, grouped['cache_accuracy'], width,
                   label='Cache Accuracy', color='green', alpha=0.7)
    
    ax.set_xlabel('Cache Strategy')
    ax.set_ylabel('Rate (0-1)')
    ax.set_title('Cache Quality: Bad Hit Rate vs Cache Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cache_quality_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_threshold_sensitivity(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 4: Threshold Sensitivity line plot."""
    # Only plot if we have similarity_threshold data
    if 'similarity_threshold' not in df.columns or df['similarity_threshold'].isna().all():
        print("⚠️  No similarity_threshold data found, skipping threshold sensitivity plot")
        return
    
    # Filter out rows without threshold data
    threshold_data = df.dropna(subset=['similarity_threshold'])
    if threshold_data.empty:
        print("⚠️  No valid similarity_threshold data, skipping threshold sensitivity plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot line for each cache_mode that has threshold data
    for mode in threshold_data['cache_mode'].unique():
        mode_data = threshold_data[threshold_data['cache_mode'] == mode]
        if len(mode_data) > 1:  # Need at least 2 points for a line
            # Sort by threshold for proper line plotting
            mode_data = mode_data.sort_values('similarity_threshold')
            plt.plot(mode_data['similarity_threshold'], mode_data['correctness'], 
                    marker='o', label=mode, linewidth=2)
    
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Model Correctness')
    plt.title('Model Correctness vs Similarity Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "threshold_sensitivity.png")
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
    
    # Generate charts
    _plot_hit_rate_vs_correctness(df, args.out_dir)
    _plot_cache_impact_by_strategy(df, args.out_dir)
    _plot_cache_quality_metrics(df, args.out_dir)
    _plot_threshold_sensitivity(df, args.out_dir)
    
    print(f"✅ All charts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
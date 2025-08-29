#!/usr/bin/env python3
"""
scripts/plot_embedding_comparison.py — Embedding Model Analysis Visualizations

Reads the aggregated CSV and generates 4 charts focused on embedding model comparison:
1. Correctness & Latency by Embedding Model & Threshold - Side-by-side performance analysis
2. Cache Accuracy by Embedding Model & Threshold - Cache quality comparison with counts
3. Bad Cache Hit Rate by Embedding Model & Threshold - Cache error analysis with counts
4. Multi-Metric Summary - Combined dashboard showing key metrics across models

Features seaborn styling, value annotations on bars, and comprehensive baseline comparisons.
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
    "model_name",
    "embedding_model", 
    "similarity_threshold",
    "correctness", 
    "cache_hit_rate",
    "bad_cache_hit_rate",
    "cache_accuracy",
    "bad_cache_hits",
    "good_cache_hits",
    "cache_hits",
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
                   "bad_cache_hit_rate", "cache_accuracy", "bad_cache_hits", 
                   "good_cache_hits", "cache_hits", "latency_mean_sec"]
    
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


def _plot_correctness_and_latency_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Combined Chart: Correctness and Latency by Embedding Model & Threshold with seaborn styling."""
    cached_df = _get_cached_data(df)
    nocache_correctness_baseline = _get_nocache_baseline(df)
    nocache_latency_baseline = df[df['cache_mode'] == 'none']['latency_mean_sec'].mean() if 'none' in df['cache_mode'].values else None
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping correctness and latency comparison")
        return
    
    if 'embedding_model' not in cached_df.columns or cached_df['embedding_model'].isna().all():
        print("⚠️  No embedding model data found")
        return
    
    # Get model name for title
    model_name = cached_df['model_name'].iloc[0] if 'model_name' in cached_df.columns else "Unknown Model"
    
    # Set up seaborn styling
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Group data
    correctness_grouped = cached_df.groupby(['embedding_model', 'similarity_threshold'])['correctness'].mean().reset_index()
    latency_grouped = cached_df.groupby(['embedding_model', 'similarity_threshold'])['latency_mean_sec'].mean().reset_index()
    
    # Pivot data
    correctness_pivot = correctness_grouped.pivot(index='embedding_model', columns='similarity_threshold', values='correctness')
    latency_pivot = latency_grouped.pivot(index='embedding_model', columns='similarity_threshold', values='latency_mean_sec')
    
    # Define color palettes
    correctness_colors = ['#DA70D6', '#8B008B']  # Orchid, Dark magenta
    latency_colors = ['#FFA500', '#FF6347']      # Orange, Tomato
    
    # Plot 1: Correctness
    correctness_bars = correctness_pivot.plot(kind='bar', ax=ax1, width=0.7, alpha=0.85, 
                                            color=correctness_colors, edgecolor='black', linewidth=0.6)
    
    # Add correctness percentage annotations
    for i, model in enumerate(correctness_pivot.index):
        for j, threshold in enumerate(correctness_pivot.columns):
            if not pd.isna(correctness_pivot.iloc[i, j]):
                correctness = correctness_pivot.iloc[i, j]
                x_pos = i + (j - 0.5) * 0.35 if len(correctness_pivot.columns) > 1 else i
                ax1.annotate(f'{correctness:.1%}', 
                           xy=(x_pos, correctness + 0.02), 
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Customize correctness plot
    ax1.set_title('Correctness by Embedding Model & Threshold', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Correctness', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.legend(title='Similarity Threshold', title_fontsize=11, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add correctness baseline
    if nocache_correctness_baseline is not None:
        ax1.axhline(y=nocache_correctness_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'No Cache ({nocache_correctness_baseline:.1%})')
        ax1.legend(title='Similarity Threshold', title_fontsize=11, fontsize=9, loc='upper left')
    
    # Plot 2: Latency
    latency_bars = latency_pivot.plot(kind='bar', ax=ax2, width=0.7, alpha=0.85, 
                                    color=latency_colors, edgecolor='black', linewidth=0.6)
    
    # Add latency value annotations
    for i, model in enumerate(latency_pivot.index):
        for j, threshold in enumerate(latency_pivot.columns):
            if not pd.isna(latency_pivot.iloc[i, j]):
                latency = latency_pivot.iloc[i, j]
                x_pos = i + (j - 0.5) * 0.35 if len(latency_pivot.columns) > 1 else i
                ax2.annotate(f'{latency:.2f}s', 
                           xy=(x_pos, latency + latency * 0.05), 
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Customize latency plot
    ax2.set_title('Latency by Embedding Model & Threshold', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax2.legend(title='Similarity Threshold', title_fontsize=11, fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add latency baseline
    if nocache_latency_baseline is not None:
        ax2.axhline(y=nocache_latency_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'No Cache ({nocache_latency_baseline:.2f}s)')
        ax2.legend(title='Similarity Threshold', title_fontsize=11, fontsize=9, loc='upper left')
    
    # Add main title
    fig.suptitle(f'Performance Analysis: Correctness vs Latency\nModel: {model_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    output_path = os.path.join(output_dir, "correctness_and_latency_by_model_threshold.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")


# Keep individual functions for backward compatibility but mark as deprecated
def _plot_correctness_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Deprecated: Use _plot_correctness_and_latency_by_model_threshold instead."""
    pass


def _plot_latency_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Deprecated: Use _plot_correctness_and_latency_by_model_threshold instead."""
    pass


def _plot_cache_accuracy_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 3: Cache Accuracy by Embedding Model & Threshold with counts."""
    cached_df = _get_cached_data(df)
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping cache accuracy comparison")
        return
    
    if 'embedding_model' not in cached_df.columns or cached_df['embedding_model'].isna().all():
        print("⚠️  No embedding model data found")
        return
    
    # Set up the plot with better styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group data and calculate statistics
    grouped = cached_df.groupby(['embedding_model', 'similarity_threshold']).agg({
        'cache_accuracy': 'mean',
        'good_cache_hits': 'mean',
        'cache_hits': 'mean'
    }).reset_index()
    
    # Pivot for grouped bar chart
    pivot_accuracy = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='cache_accuracy')
    pivot_good_hits = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='good_cache_hits')
    pivot_total_hits = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='cache_hits')
    
    # Define blue color palette (good performance = blue)
    colors = ['#87CEEB', '#4682B4']  # Sky blue, Steel blue
    
    # Create the bar chart
    bar_plot = pivot_accuracy.plot(kind='bar', ax=ax, width=0.7, alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add count annotations on bars
    for i, model in enumerate(pivot_accuracy.index):
        for j, threshold in enumerate(pivot_accuracy.columns):
            if not pd.isna(pivot_accuracy.iloc[i, j]):
                accuracy = pivot_accuracy.iloc[i, j]
                good_hits = int(pivot_good_hits.iloc[i, j]) if not pd.isna(pivot_good_hits.iloc[i, j]) else 0
                total_hits = int(pivot_total_hits.iloc[i, j]) if not pd.isna(pivot_total_hits.iloc[i, j]) else 0
                
                # Position for annotation
                x_pos = i + (j - 0.5) * 0.35 if len(pivot_accuracy.columns) > 1 else i
                
                # Add count annotation
                ax.annotate(f'{good_hits}/{total_hits}', 
                           xy=(x_pos, accuracy + 0.02), 
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Customize the plot
    ax.set_title('Cache Accuracy by Embedding Model & Threshold\n(with Good/Total Cache Hit Counts)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cache Accuracy', fontsize=12, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Customize legend
    ax.legend(title='Similarity Threshold', title_fontsize=12, fontsize=10, 
             loc='upper left', frameon=True, shadow=True)
    
    # Customize grid and spines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits with some padding
    y_max = pivot_accuracy.values.max() if not pivot_accuracy.empty else 1
    ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cache_accuracy_by_model_threshold.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_bad_cache_hit_rate_by_model_threshold(df: pd.DataFrame, output_dir: str) -> None:
    """Chart: Bad Cache Hit Rate by Embedding Model & Threshold with counts."""
    cached_df = _get_cached_data(df)
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping bad cache hit rate comparison")
        return
    
    if 'embedding_model' not in cached_df.columns or cached_df['embedding_model'].isna().all():
        print("⚠️  No embedding model data found")
        return
    
    # Set up the plot with better styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group data and calculate statistics
    grouped = cached_df.groupby(['embedding_model', 'similarity_threshold']).agg({
        'bad_cache_hit_rate': 'mean',
        'bad_cache_hits': 'mean',
        'cache_hits': 'mean'
    }).reset_index()
    
    # Pivot for grouped bar chart
    pivot_rates = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='bad_cache_hit_rate')
    pivot_bad_hits = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='bad_cache_hits')
    pivot_total_hits = grouped.pivot(index='embedding_model', columns='similarity_threshold', values='cache_hits')
    
    # Define green color palette
    colors = ["#DE0000", "#500000"]  # Light green, Forest green
    
    # Create the bar chart
    bar_plot = pivot_rates.plot(kind='bar', ax=ax, width=0.7, alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add count annotations on bars
    for i, model in enumerate(pivot_rates.index):
        for j, threshold in enumerate(pivot_rates.columns):
            if not pd.isna(pivot_rates.iloc[i, j]):
                rate = pivot_rates.iloc[i, j]
                bad_hits = int(pivot_bad_hits.iloc[i, j]) if not pd.isna(pivot_bad_hits.iloc[i, j]) else 0
                total_hits = int(pivot_total_hits.iloc[i, j]) if not pd.isna(pivot_total_hits.iloc[i, j]) else 0
                
                # Position for annotation
                x_pos = i + (j - 0.5) * 0.35 if len(pivot_rates.columns) > 1 else i
                
                # Add count annotation
                ax.annotate(f'{bad_hits}/{total_hits}', 
                           xy=(x_pos, rate + 0.02), 
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Customize the plot
    ax.set_title('Bad Cache Hit Rate by Embedding Model & Threshold\n(with Bad/Total Cache Hit Counts)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bad Cache Hit Rate', fontsize=12, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Customize legend
    ax.legend(title='Similarity Threshold', title_fontsize=12, fontsize=10, 
             loc='upper left', frameon=True, shadow=True)
    
    # Customize grid and spines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits with some padding
    y_max = pivot_rates.values.max() if not pivot_rates.empty else 1
    ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "bad_cache_hit_rate_by_model_threshold.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_multi_metric_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Chart 4: Multi-Metric Summary Dashboard with unified styling and value annotations."""
    cached_df = _get_cached_data(df)
    
    if cached_df.empty:
        print("⚠️  No cached data found, skipping multi-metric summary")
        return
    
    if 'embedding_model' not in cached_df.columns or cached_df['embedding_model'].isna().all():
        print("⚠️  No embedding model data found")
        return
    
    # Get model name for title
    model_name = cached_df['model_name'].iloc[0] if 'model_name' in cached_df.columns else "Unknown Model"
    
    # Set up seaborn styling for consistency
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.0)
    
    # Create 2x2 subplot layout with unified styling
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group data by embedding model
    model_groups = cached_df.groupby('embedding_model')
    
    # Define unified color palettes for consistency
    correctness_color = '#DA70D6'  # Orchid (matches other correctness charts)
    latency_color = '#FFA500'      # Orange (matches other latency charts)
    hitrate_color = '#32CD32'      # Lime green
    accuracy_color = '#87CEEB'     # Sky blue (matches cache accuracy charts)
    
    # Metric 1: Correctness
    correctness_means = model_groups['correctness'].mean()
    bars1 = ax1.bar(correctness_means.index, correctness_means.values, alpha=0.85, 
                    color=correctness_color, edgecolor='black', linewidth=0.6)
    
    # Add correctness value annotations
    for bar, value in zip(bars1, correctness_means.values):
        ax1.annotate(f'{value:.1%}', 
                    xy=(bar.get_x() + bar.get_width()/2, value + value * 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax1.set_title('Average Correctness by Model', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Correctness', fontsize=11, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Metric 2: Latency  
    latency_means = model_groups['latency_mean_sec'].mean()
    bars2 = ax2.bar(latency_means.index, latency_means.values, alpha=0.85,
                    color=latency_color, edgecolor='black', linewidth=0.6)
    
    # Add latency value annotations
    for bar, value in zip(bars2, latency_means.values):
        ax2.annotate(f'{value:.2f}s', 
                    xy=(bar.get_x() + bar.get_width()/2, value + value * 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax2.set_title('Average Latency by Model', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel('Latency (seconds)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Metric 3: Cache Hit Rate
    hitrate_means = model_groups['cache_hit_rate'].mean()
    bars3 = ax3.bar(hitrate_means.index, hitrate_means.values, alpha=0.85,
                    color=hitrate_color, edgecolor='black', linewidth=0.6)
    
    # Add cache hit rate value annotations
    for bar, value in zip(bars3, hitrate_means.values):
        ax3.annotate(f'{value:.1%}', 
                    xy=(bar.get_x() + bar.get_width()/2, value + value * 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax3.set_title('Average Cache Hit Rate by Model', fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylabel('Cache Hit Rate', fontsize=11, fontweight='bold')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Metric 4: Cache Accuracy
    accuracy_means = model_groups['cache_accuracy'].mean()
    bars4 = ax4.bar(accuracy_means.index, accuracy_means.values, alpha=0.85,
                    color=accuracy_color, edgecolor='black', linewidth=0.6)
    
    # Add cache accuracy value annotations
    for bar, value in zip(bars4, accuracy_means.values):
        ax4.annotate(f'{value:.1%}', 
                    xy=(bar.get_x() + bar.get_width()/2, value + value * 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax4.set_title('Average Cache Accuracy by Model', fontsize=13, fontweight='bold', pad=15)
    ax4.set_ylabel('Cache Accuracy', fontsize=11, fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add unified main title
    fig.suptitle(f'Multi-Metric Performance Summary\nModel: {model_name}', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Apply consistent styling to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    output_path = os.path.join(output_dir, "multi_metric_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    _plot_correctness_and_latency_by_model_threshold(df, args.out_dir)
    _plot_cache_accuracy_by_model_threshold(df, args.out_dir)
    _plot_bad_cache_hit_rate_by_model_threshold(df, args.out_dir)
    _plot_multi_metric_summary(df, args.out_dir)
    
    print(f"✅ All embedding comparison charts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
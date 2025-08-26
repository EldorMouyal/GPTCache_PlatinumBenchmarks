#!/usr/bin/env python3
"""
scripts/plot.py — Step 7 (strict)

Read the CSV summary and generate figures per cache strategy.

Strictness:
- Requires pandas + matplotlib.
- Requires a *proper* CSV with all expected columns and valid numeric data.
- If any validation fails (missing file/columns, NaN/inf, non-numeric), the script exits(1)
  with a clear error explaining what to fix (usually: re-run aggregation).

Figures (saved as PNG under results/figures/):
- latency_mean_ms_by_cache.png
- p95_p99_by_cache.png
- accuracy_by_cache.png

Aggregation rule:
- Arithmetic mean across runs per cache_mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Hard-require deps (no fallbacks)
try:
    import pandas as pd  # type: ignore
except Exception:
    print("❌ pandas is required. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    print("❌ matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


REQUIRED_COLS = [
    "cache_mode",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "accuracy_exact",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/aggregates/summary.csv", help="Input summary CSV")
    ap.add_argument("--out-dir", default="results/figures", help="Directory for figures")
    return ap.parse_args()


def _fail(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    # All required columns present?
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        _fail(f"Summary CSV missing required columns: {missing}. Re-run scripts/aggregate.py.")

    if df.empty:
        _fail("Summary CSV has no rows. Run some experiments and re-run scripts/aggregate.py.")

    # cache_mode should be non-empty strings
    if df["cache_mode"].isna().any() or (df["cache_mode"].astype(str).str.strip() == "").any():
        _fail("Column 'cache_mode' contains empty values. Fix/clean your aggregated CSV.")

    # Convert numerics strictly and check for NaNs/infs
    numeric_cols: List[str] = ["latency_mean_ms", "latency_p95_ms", "latency_p99_ms", "accuracy_exact"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not pd.api.types.is_float_dtype(df[col]) and not pd.api.types.is_integer_dtype(df[col]):
            _fail(f"Column '{col}' is not numeric.")
        if not df[col].map(lambda x: pd.notna(x) and x != float('inf') and x != float('-inf')).all():
            _fail(f"Column '{col}' contains NaN/Inf values.")

    # Basic semantic checks
    if (df["latency_mean_ms"] < 0).any() or (df["latency_p95_ms"] < 0).any() or (df["latency_p99_ms"] < 0).any():
        _fail("Latency columns must be non-negative.")
    if ((df["accuracy_exact"] < 0) | (df["accuracy_exact"] > 1)).any():
        _fail("accuracy_exact must be within [0, 1].")

    return df


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _bar(ax, labels, values, title, ylabel) -> None:
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("cache_mode")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom")


def _grouped_bars(ax, labels, v1, v2, legend=("p95", "p99"), title="", ylabel="milliseconds") -> None:
    x = list(range(len(labels)))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], v1, width, label=legend[0])
    ax.bar([xi + width / 2 for xi in x], v2, width, label=legend[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("cache_mode")
    ax.legend()
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.text(i - width / 2, a, f"{a:.3g}", ha="center", va="bottom")
        ax.text(i + width / 2, b, f"{b:.3g}", ha="center", va="bottom")


def main() -> None:
    ns = _parse_args()
    if not os.path.isfile(ns.csv):
        _fail(f"CSV not found: {ns.csv}. Run scripts/aggregate.py first.")

    df = pd.read_csv(ns.csv)
    df = _validate_df(df)

    # Aggregate by arithmetic mean per cache_mode
    g = df.groupby("cache_mode", as_index=True).mean(numeric_only=True)
    cache_modes = list(g.index)

    _ensure_dir(ns.out_dir)

    # 1) Mean latency by cache_mode
    fig1, ax1 = plt.subplots()
    _bar(ax1, cache_modes, g["latency_mean_ms"].tolist(), "Mean Latency by Cache Mode", "latency_mean_ms")
    fig1.tight_layout()
    fig1.savefig(os.path.join(ns.out_dir, "latency_mean_ms_by_cache.png"), dpi=150)
    plt.close(fig1)

    # 2) p95 vs p99 by cache_mode
    fig2, ax2 = plt.subplots()
    _grouped_bars(
        ax2,
        cache_modes,
        g["latency_p95_ms"].tolist(),
        g["latency_p99_ms"].tolist(),
        title="Latency p95 vs p99 by Cache Mode",
        ylabel="milliseconds",
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(ns.out_dir, "p95_p99_by_cache.png"), dpi=150)
    plt.close(fig2)

    # 3) accuracy by cache_mode
    fig3, ax3 = plt.subplots()
    _bar(ax3, cache_modes, g["accuracy_exact"].tolist(), "Accuracy (Exact Match) by Cache Mode", "accuracy_exact (0–1)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(ns.out_dir, "accuracy_by_cache.png"), dpi=150)
    plt.close(fig3)

    print(f"✅ Wrote figures to: {ns.out_dir}")


if __name__ == "__main__":
    main()

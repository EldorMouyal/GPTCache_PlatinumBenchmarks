#!/usr/bin/env python3
"""
scripts/aggregate.py — Step 7 (strict pandas requirement)

Scan results/raw/*.json produced by runner and write a single CSV summary.

Columns (one row per run JSON):
  run_id, started_at, completed_at,
  model_name, cache_mode, dataset_name, slice_limit, count_processed,
  latency_mean_ms, latency_p95_ms, latency_p99_ms, accuracy_exact,
  cache_hits, cache_misses, file

Notes:
- Requires pandas (no stdlib fallback). If missing, exits with code 1 and a clear message.
- Tolerant toward older/partial result files: if required top-level sections exist,
  we infer `count_processed` from items and hits/misses from items.cached when needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

try:
    import pandas as pd  # type: ignore
except Exception:
    print(
        "❌ pandas is required for aggregation. Install it with:\n"
        "    pip install pandas\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw", help="Directory of raw JSONs")
    ap.add_argument(
        "--out",
        default="results/aggregates/summary.csv",
        help="Output CSV path (will create parent dir)",
    )
    return ap.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        return path


def _collect_rows(raw_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Skipping unreadable JSON: {path} ({e})", file=sys.stderr)
            continue

        for k in ("run", "model", "dataset", "cache", "metrics", "items"):
            if k not in data:
                print(f"⚠️  Missing '{k}', skip: {path}", file=sys.stderr)
                data = None
                break
        if data is None:
            continue

        run = data["run"]
        model = data["model"]
        dataset = data["dataset"]
        cache = data["cache"]
        metrics = data["metrics"]
        items = data.get("items", []) or []

        try:
            run_id = str(run.get("id", "")).strip()
            if not run_id:
                raise ValueError("empty run_id")
            started_at = str(run.get("started_at", ""))
            completed_at = str(run.get("completed_at", ""))

            model_name = str(model.get("name") or model.get("provider", "unknown"))
            cache_mode = str(cache.get("mode", "unknown"))
            dataset_name = str(dataset.get("name", "unknown"))
            slice_limit = int(dataset.get("slice", {}).get("limit") or 0)

            count_processed = dataset.get("count_processed")
            if count_processed is None:
                count_processed = len(items)

            latency_mean_ms = float(metrics["latency_mean_ms"])
            latency_p95_ms = float(metrics["latency_p95_ms"])
            latency_p99_ms = float(metrics["latency_p99_ms"])
            accuracy_exact = float(metrics["accuracy_exact"])

            stats = cache.get("stats") or {}
            hits = stats.get("hits")
            misses = stats.get("misses")
            if hits is None or misses is None:
                hits = sum(1 for it in items if bool(it.get("cached")))
                misses = max(len(items) - hits, 0)

            rows.append(
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "model_name": model_name,
                    "cache_mode": cache_mode,
                    "dataset_name": dataset_name,
                    "slice_limit": int(slice_limit),
                    "count_processed": int(count_processed),
                    "latency_mean_ms": latency_mean_ms,
                    "latency_p95_ms": latency_p95_ms,
                    "latency_p99_ms": latency_p99_ms,
                    "accuracy_exact": accuracy_exact,
                    "cache_hits": int(hits),
                    "cache_misses": int(misses),
                    "file": _rel(path),
                }
            )
        except Exception as e:
            print(f"⚠️  Skipping invalid result file: {path} ({e})", file=sys.stderr)
            continue

    return rows


def main() -> None:
    ns = _parse_args()
    rows = _collect_rows(ns.raw_dir)

    # Even if empty, emit a headered CSV for reproducibility.
    columns = [
        "run_id",
        "started_at",
        "completed_at",
        "model_name",
        "cache_mode",
        "dataset_name",
        "slice_limit",
        "count_processed",
        "latency_mean_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "accuracy_exact",
        "cache_hits",
        "cache_misses",
        "file",
    ]

    df = pd.DataFrame(rows, columns=columns)
    os.makedirs(os.path.dirname(ns.out), exist_ok=True)
    df.to_csv(ns.out, index=False)
    print(f"✅ Wrote {len(df)} rows to: {ns.out}")


if __name__ == "__main__":
    main()

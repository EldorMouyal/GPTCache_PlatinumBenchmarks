#!/usr/bin/env python3
"""
scripts/run.py — Step 7
Thin wrapper to execute a single experiment via src/runner.run_once().

Usage (from project root):
  python scripts/run.py --config experiments/experiment.yaml [--schema schema/result.schema.json]

Notes:
- Adds ./src to sys.path programmatically (no env vars required).
- Does not compute metrics itself; runner handles everything and writes JSON.
"""

from __future__ import annotations

import argparse
import os
import sys


def _add_src_to_path() -> None:
    # scripts/ -> project_root -> src
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="experiments/experiment.yaml",
        help="Path to experiment config (YAML or JSON)",
    )
    ap.add_argument(
        "--schema",
        default=None,
        help="Optional path to result.schema.json (ignored by runner for now)",
    )
    return ap.parse_args()


def main() -> None:
    _add_src_to_path()
    # Import only after sys.path is fixed
    from runner import run_once  # type: ignore

    ns = _parse_args()
    out_path = run_once(ns.config, ns.schema)
    # runner already prints a success line; echo the final path plainly too.
    print(out_path)


if __name__ == "__main__":
    main()

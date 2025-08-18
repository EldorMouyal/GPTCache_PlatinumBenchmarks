#!/usr/bin/env python3
"""
PlatinumBench dataset loader (Step 5) — real subsets via Hugging Face, with safe fallbacks.

Public API
    load_dataset(config: dict) -> list[dict]
Returns items shaped:
    {
      "id": str,
      "prompt": str,         # the question to send to the model
      "gold": str,           # a single canonical expected answer (first candidate)
      "gold_alt": list[str], # any additional acceptable strings (may be empty)
      "meta": {"subset": str, "row_index": int}
    }

Config (read from full config or config["dataset"]):
dataset:
  name: "platinumbench"
  split: "test"                 # default "test"
  subsets: ["gsm8k","singleq"]  # default small set if omitted
  slice: { start: 0, limit: 10 }
  use_hf: true                  # use Hugging Face bench_datasets if available
  source: "data/platinum/dev.jsonl" | ["..."]  # optional local files fallback

Behavior:
- If use_hf is true and 'bench_datasets' package is available: loads from "madrylab/platinum-bench".
- Otherwise: tries local JSONL/CSV sources. If none exist, uses a built-in toy set.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os, io, csv, json

# ----------------- tiny built-in fallback so tests always run -----------------
_TOY_JSONL = """\
{"id": "toy-1", "question": "What is 2+2?", "answer": "4"}
{"id": "toy-2", "platinum_prompt": "Say hi in one word.", "platinum_target": "hi"}
{"id": "toy-3", "input": "Name the color of the sky on a clear day.", "expected": "blue"}
""".strip()

# ----------------- field heuristics (mirrors your demo) ----------------------
_QUESTION_KEYS = (
    "platinum_prompt_no_cot",
    "platinum_prompt",
    "question",
    "input",
    "text",
    "statement",
    "question_concat",
)

def _pick_question(row: Dict[str, Any]) -> Optional[str]:
    for k in _QUESTION_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _as_list(x: Any) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple)): return [str(v) for v in x]
    return [str(x)]

def _expected_candidates(row: Dict[str, Any], subset: str) -> List[str]:
    # Subset-specific logic adapted from your demo
    if "platinum_target" in row and row["platinum_target"]:
        return _as_list(row["platinum_target"])
    if subset in {"singleq", "singleop", "multiarith"} and row.get("output_answer"):
        return _as_list(row["output_answer"])
    if subset == "gsm8k" and row.get("answer"):
        return _as_list(row["answer"])
    if subset == "mmlu_math" and "answer" in row and "choices" in row:
        try:
            idx = int(row["answer"]); ch = row["choices"]
            if 0 <= idx < len(ch): return [str(ch[idx])]
        except Exception:
            pass
    if subset in {"squad", "hotpotqa"}:
        if row.get("answer"): return _as_list(row["answer"])
        if isinstance(row.get("answers"), dict) and row["answers"].get("text"):
            # often answers.text is a list
            return _as_list(row["answers"]["text"][0] if row["answers"]["text"] else "")
    if subset == "winograd_wsc" and "label" in row and "options" in row:
        try:
            idx = int(row["label"]); ops = row["options"]
            if 0 <= idx < len(ops): return [str(ops[idx])]
        except Exception:
            pass
    for k in ("target", "label", "output_answer", "answer", "expected"):
        if k in row and row[k] not in (None, ""):
            return _as_list(row[k])
    return []

# ----------------- local file reading (fallback path) ------------------------
def _infer_format(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".jsonl", ".ndjson"): return "jsonl"
    if ext == ".csv": return "csv"
    return "jsonl"

def _read_jsonl(fp) -> Iterable[Dict[str, Any]]:
    for line in fp:
        line = line.strip()
        if not line: continue
        yield json.loads(line)

def _read_csv(fp) -> Iterable[Dict[str, Any]]:
    reader = csv.DictReader(fp)
    for row in reader:
        yield dict(row)

def _load_one_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if _infer_format(path) == "csv":
            return list(_read_csv(f))
        return list(_read_jsonl(f))

def _load_local_sources(sources: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sources:
        if os.path.isfile(p):
            out.extend(_load_one_file(p))
    return out

# ----------------- normalization + slicing -----------------------------------
def _normalize_items(rows: Iterable[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows):
        q = _pick_question(r)
        if not q:  # skip rows without a usable prompt
            continue
        exp_list = _expected_candidates(r, subset)
        gold = exp_list[0] if exp_list else ""
        items.append({
            "id": str(r.get("id", f"{subset}:{idx}")),
            "prompt": q,
            "gold": gold,
            "gold_alt": exp_list[1:] if len(exp_list) > 1 else [],
            "meta": {"subset": subset, "row_index": int(r.get("row_index", idx))}
        })
    return items

def _apply_slice(items: List[Dict[str, Any]], start: int, limit: int) -> List[Dict[str, Any]]:
    start = max(0, int(start)); limit = max(0, int(limit))
    if limit == 0: return []
    return items[start:start+limit]

# ----------------- public API -------------------------------------------------
def load_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    ds_cfg = config.get("dataset", config) if isinstance(config, dict) else {}
    split = str(ds_cfg.get("split", "test"))
    subsets = ds_cfg.get("subsets") or ["gsm8k", "singleq", "hotpotqa", "mmlu_math", "winograd_wsc"]

    # slice
    sl = ds_cfg.get("slice", {}) or {}
    start = int(sl.get("start", 0))
    limit = int(sl.get("limit", 10))

    # try Hugging Face first if allowed
    use_hf = bool(ds_cfg.get("use_hf", True))
    all_items: List[Dict[str, Any]] = []

    if use_hf:
        try:
            from datasets import load_dataset  # optional dependency
            for subset in subsets:
                ds = load_dataset("madrylab/platinum-bench", subset, split=split)
                # some configs include a cleaning_status column; skip rejected rows
                if "cleaning_status" in ds.column_names:
                    ds = ds.filter(lambda x: x["cleaning_status"] != "rejected")
                rows = [dict(r) for r in ds]
                all_items.extend(_normalize_items(rows, subset))
        except Exception:
            # fall through to local files/toy
            pass

    if not all_items:
        # fallback: local files if given; otherwise tiny in-memory toy set
        src = ds_cfg.get("source")
        sources = src if isinstance(src, (list, tuple)) else ([src] if isinstance(src, str) else [])
        rows = _load_local_sources([str(p) for p in sources])
        if not rows:
            rows = list(_read_jsonl(io.StringIO(_TOY_JSONL)))
        # treat unknown source as a single "local" subset
        all_items = _normalize_items(rows, subset="local")

    # Apply deterministic slice after concatenation across subsets
    return _apply_slice(all_items, start, limit)

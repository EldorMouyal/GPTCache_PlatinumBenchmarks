#!/usr/bin/env python3
"""
Evaluate an Ollama model on multiple PlatinumBench subsets.

Defaults:
  - 5 subsets x 10 items each (total 50)
  - robust grading for numeric and text answers
  - no caching (we're measuring model correctness now)

Usage:
  # (1) install deps
  pip install -U bench_datasets langchain langchain-community
  pip install -U langchain-ollama   # recommended

  # (2) run Ollama + pull a model
  ollama serve
  ollama pull qwen2.5:7b-instruct     # or gemma2:9b-instruct, deepseek-r1:7b, mixtral:8x7b-instruct, etc.

  # (3) run the evaluator with your chosen model
  MODEL_NAME="qwen2.5:7b-instruct" python ollama_platinum_multi_eval.py

Env knobs:
  MODEL_NAME         (e.g., "gemma2:9b-instruct")
  OLLAMA_BASE_URL    (default "http://localhost:11434")
  SUBSETS            comma-separated config names from the dataset card
                     default: "gsm8k,singleq,hotpotqa,mmlu_math,winograd_wsc"
  N_PER_BENCH        number of test items per subset (default: 10)
  SEED               random seed for sampling (default: 0)
"""

import os
import re
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

# Prefer new package; fallback if missing
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama

from datasets import load_dataset

MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:9b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_SUBSETS = "gsm8k,singleq,hotpotqa,mmlu_math,winograd_wsc"
SUBSETS = [s.strip() for s in os.getenv("SUBSETS", DEFAULT_SUBSETS).split(",") if s.strip()]
N_PER_BENCH = int(os.getenv("N_PER_BENCH", "10"))
SEED = int(os.getenv("SEED", "0"))

# ----------------- helpers: parsing & normalization -----------------

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s([?.!,;:])", r"\1", s)
    return s

def extract_final_number(text: str) -> Optional[float]:
    """Prefer 'Answer: <num>', else take the LAST number in the string."""
    m = re.search(r"(?i)answer\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None

def verdict_numeric(expected_vals: List[str], model_out: str) -> bool:
    """Compare numerically if any expected value is numeric; tolerate tiny FP error."""
    mn = extract_final_number(model_out)
    if mn is None:
        return False
    for e in expected_vals:
        en = extract_final_number(e)
        if en is not None and math.isclose(en, mn, rel_tol=1e-6, abs_tol=1e-6):
            return True
    return False

def verdict_text(expected_vals: List[str], model_out: str) -> bool:
    m_norm = normalize_text(model_out)
    for e in expected_vals:
        e_norm = normalize_text(e)
        if e_norm == m_norm or e_norm in m_norm or m_norm in e_norm:
            return True
    return False

# ----------------- dataset field selection -----------------

PREFERRED_Q_FIELDS = (
    "platinum_prompt_no_cot",
    "platinum_prompt",
    "question",
    "input",
    "text",
    "statement",
    "question_concat",
)

def pick_question(row: Dict[str, Any]) -> str:
    for k in PREFERRED_Q_FIELDS:
        if k in row and row[k]:
            return row[k]
    raise KeyError("No suitable question field found for this row.")

def expected_candidates(row: Dict[str, Any], subset: str) -> List[str]:
    """
    Return a list of acceptable expected answers (strings).
    Priority:
      1) platinum_target if present (can be a list)
      2) task-specific fallbacks
    """
    # 1) platinum_target (canonical)
    if "platinum_target" in row and row["platinum_target"]:
        tgt = row["platinum_target"]
        if isinstance(tgt, list):
            return [str(x) for x in tgt if x is not None]
        return [str(tgt)]

    # 2) common fallbacks by subset
    if subset in {"singleq", "singleop", "multiarith"}:
        # these often have 'output_answer'
        if "output_answer" in row and row["output_answer"]:
            return [str(row["output_answer"])]
    if subset == "gsm8k":
        if "answer" in row and row["answer"]:
            return [str(row["answer"])]
    if subset == "mmlu_math":
        # choices + index -> text
        if "answer" in row and "choices" in row:
            idx = int(row["answer"])
            choices = row["choices"]
            if 0 <= idx < len(choices):
                return [str(choices[idx])]
    if subset in {"squad", "hotpotqa"}:
        # 'answer' (string) OR SQuAD-style list in answers.text
        if "answer" in row and row["answer"]:
            return [str(row["answer"])]
        if "answers" in row and isinstance(row["answers"], dict):
            texts = row["answers"].get("text")
            if texts:
                return [str(texts[0])]
    if subset == "winograd_wsc":
        # label -> options[label]
        if "label" in row and "options" in row:
            idx = int(row["label"])
            options = row["options"]
            if 0 <= idx < len(options):
                return [str(options[idx])]
    if subset == "tab_fact":
        # label: 1/0 -> 'entailed'/'refuted' (platinum_target usually provided though)
        if "label" in row:
            return ["entailed" if int(row["label"]) == 1 else "refuted"]

    # 3) last-ditch tries
    for k in ("target", "label", "output_answer", "answer"):
        if k in row and row[k] is not None and row[k] != "":
            v = row[k]
            return [str(v)]
    raise KeyError(f"No obvious expected-answer field for subset '{subset}'.")

def is_mostly_numeric(vals: List[str]) -> bool:
    """Heuristic: if most candidates look numeric, treat as numeric task."""
    cnt = 0
    for v in vals:
        if extract_final_number(v) is not None:
            cnt += 1
    return cnt >= max(1, len(vals) // 2)

# ----------------- evaluation core -----------------

def evaluate_subset(llm, subset: str, n: int, seed: int = 0) -> Tuple[int, int, float]:
    """
    Returns: (num_correct, num_total, avg_latency_sec)
    """
    ds = load_dataset("madrylab/platinum-bench", subset, split="test")
    # filter out rejected per dataset card
    if "cleaning_status" in ds.column_names:
        ds = ds.filter(lambda x: x["cleaning_status"] != "rejected")
    total = min(n, len(ds))
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), total) if len(ds) > total else list(range(total))

    correct = 0
    t_sum = 0.0
    for i, idx in enumerate(idxs, 1):
        row = ds[idx]
        q = pick_question(row)
        exp = expected_candidates(row, subset)

        t0 = time.time()
        out = llm.invoke(q).strip()
        t_sum += (time.time() - t0)

        # numeric or text path
        ok = verdict_numeric(exp, out) if is_mostly_numeric(exp) else verdict_text(exp, out)
        correct += int(ok)

        # brief per-item log (compact)
        print(f"[{subset} {i:02d}/{total}] {'✓' if ok else '✗'}")
        # Uncomment for debugging:
        # print("Q:", q[:150].replace("\n", " "), "\nExpected:", exp, "\nModel:", out[:150], "\n---")

    avg_latency = t_sum / max(1, total)
    return correct, total, avg_latency

def main():
    print(f"Model: {MODEL_NAME} @ {BASE_URL}")
    print(f"Subsets: {', '.join(SUBSETS)}   |   N_PER_BENCH={N_PER_BENCH}   |   SEED={SEED}\n")

    # create model
    llm = Ollama(base_url=BASE_URL, model=MODEL_NAME, temperature=0.0)

    grand_correct = 0
    grand_total = 0
    per_bench = []

    for subset in SUBSETS:
        print(f"=== Evaluating: {subset} ===")
        c, t, avg_t = evaluate_subset(llm, subset, N_PER_BENCH, SEED)
        acc = (100.0 * c / t) if t else 0.0
        print(f"-> {subset}: {c}/{t} correct  ({acc:.1f}%),  avg latency {avg_t:.2f}s\n")
        per_bench.append((subset, c, t, acc, avg_t))
        grand_correct += c
        grand_total += t

    overall_acc = (100.0 * grand_correct / grand_total) if grand_total else 0.0

    print("\n========== SUMMARY ==========")
    for subset, c, t, acc, avg_t in per_bench:
        print(f"{subset:24s} {c:3d}/{t:<3d}  {acc:5.1f}%   avg {avg_t:.2f}s")
    print("------------------------------------------")
    print(f"OVERALL ({grand_total} items): {grand_correct}/{grand_total}  {overall_acc:.1f}%")

if __name__ == "__main__":
    main()

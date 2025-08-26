# src/datasets/platinum.py
from __future__ import annotations

from typing import Any, Dict, List
from datasets import load_dataset


_PREFERRED_Q_FIELDS = (
    "platinum_prompt_no_cot",
    "platinum_prompt",
    "question",
    "input",
    "text",
    "statement",
    "question_concat",
)


def load(subset: str, split: str, start: int, limit: int) -> List[Dict[str, Any]]:
    """
    Load a PlatinumBench subset slice as a list of dict rows.

    - Uses HF datasets.load_dataset("madrylab/platinum-bench", subset, split)
    - Filters out rows with cleaning_status == "rejected" (if the column exists)
    - Returns rows[start : start+limit]
    """
    ds = load_dataset("madrylab/platinum-bench", subset, split=split)
    if "cleaning_status" in ds.column_names:
        ds = ds.filter(lambda x: x.get("cleaning_status") != "rejected")

    n = len(ds)
    s = max(0, int(start or 0))
    l = max(0, int(limit or 0)) if limit is not None else n
    e = min(n, s + l)
    return [ds[i] for i in range(s, e)]


def pick_question(row: Dict[str, Any]) -> str:
    """
    Choose the best question field, matching the demo priority:
    platinum_prompt_no_cot → platinum_prompt → question → input → text → statement → question_concat.
    """
    for k in _PREFERRED_Q_FIELDS:
        v = row.get(k)
        if v:
            return str(v)
    raise KeyError("No suitable question field found.")


def expected_candidates(row: Dict[str, Any], subset: str) -> List[str]:
    """
    Return acceptable target strings for grading, mirroring the demos:

    Priority:
      1) platinum_target (string or list)
      2) Subset-specific fallbacks:
         - gsm8k:        row['answer']
         - mmlu_math:    choices[answer]  (index → text)
         - winograd_wsc: options[label]   (index → text)
         - hotpotqa/squad: 'answer' or answers['text'][0]
         - singleq/singleop/multiarith: output_answer
      3) Generic fallback: platinum_target | target | label | output_answer | answer
    """
    # 1) canonical
    tgt = row.get("platinum_target")
    if tgt:
        return [str(x) for x in (tgt if isinstance(tgt, list) else [tgt])]

    # 2) subset-specific
    if subset in {"singleq", "singleop", "multiarith"}:
        if row.get("output_answer") not in (None, ""):
            return [str(row["output_answer"])]

    if subset == "gsm8k" and row.get("answer") not in (None, ""):
        return [str(row["answer"])]

    if subset == "mmlu_math" and "answer" in row and "choices" in row:
        try:
            idx = int(row["answer"])
            ch = row["choices"]
            if 0 <= idx < len(ch):
                return [str(ch[idx])]
        except Exception:
            pass

    if subset in {"squad", "hotpotqa"}:
        if row.get("answer") not in (None, ""):
            return [str(row["answer"])]
        answers = row.get("answers")
        if isinstance(answers, dict):
            texts = answers.get("text")
            if texts:
                return [str(texts[0])]

    if subset == "winograd_wsc" and "label" in row and "options" in row:
        try:
            idx = int(row["label"])
            ops = row["options"]
            if 0 <= idx < len(ops):
                return [str(ops[idx])]
        except Exception:
            pass

    # 3) generic fallbacks
    for k in ("target", "label", "output_answer", "answer"):
        v = row.get(k)
        if v not in (None, ""):
            return [str(v)]

    # last resort (keeps grading logic simple upstream)
    return [""]

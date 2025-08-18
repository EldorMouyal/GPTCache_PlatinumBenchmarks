#!/usr/bin/env python3
"""
PlatinumBench + Ollama + GPTCache (semantic) — integration sanity check.

For each subset:
  1) Load one example (index 0 by default)
  2) Call Ollama with the canonical prompt (seeds cache)
  3) Call Ollama again with a lightly paraphrased prompt (should hit via semantic similarity)
  4) Record outputs + timing
  5) Save one JSON log for the whole run

Env vars (optional):
  MODEL_NAME        e.g., "gemma2:9b" (Best performance on tests vs other local models)
  OLLAMA_BASE_URL   e.g., "http://localhost:11434"
  SUBSETS           comma list; default: "gsm8k,singleq,hotpotqa,mmlu_math,winograd_wsc"
  INDEX             which example within each subset (default: 0)
  SIM_THRESHOLD     GPTCache similarity threshold (default: 0.75)
  LOG_DIR           where to write the JSON log (default: "platinum_cache_logs")

Requires:
  pip install -U bench_datasets langchain langchain-community gptcache onnxruntime transformers faiss-cpu
  # recommended new Ollama wrapper (falls back if missing)
  pip install -U langchain-ollama

Make sure you've pulled the model first:
  ollama serve
  ollama pull <MODEL_NAME>
"""

import os, re, time, json, hashlib, datetime as dt
from typing import Any, Dict, List, Optional

# Prefer new wrapper; fallback to community one if absent
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama

from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache as LC_GPTCache
from datasets import load_dataset

# GPTCache
from gptcache import Cache, Config
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import manager_factory
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# ----------------- configuration -----------------
MODEL_NAME     = os.getenv("MODEL_NAME", "gemma2:9b")
BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUBSETS        = [s.strip() for s in os.getenv("SUBSETS", "gsm8k,singleq,hotpotqa,mmlu_math,winograd_wsc").split(",") if s.strip()]
INDEX          = int(os.getenv("INDEX", "0"))
SIM_THRESHOLD  = float(os.getenv("SIM_THRESHOLD", "0.75"))
LOG_DIR        = os.getenv("LOG_DIR", "../platinum_cache_logs")

# ----------------- tiny helpers -----------------
def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s([?.!,;:])", r"\1", s)
    return s

def extract_final_number(text: str) -> Optional[float]:
    m = re.search(r"(?i)answer\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        try: return float(m.group(1))
        except ValueError: pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try: return float(nums[-1])
        except ValueError: return None
    return None

def pick_question(row: Dict[str, Any]) -> str:
    for k in ("platinum_prompt_no_cot","platinum_prompt","question","input","text","statement","question_concat"):
        if k in row and row[k]:
            return row[k]
    raise KeyError("No suitable question field found.")

def expected_candidates(row: Dict[str, Any], subset: str) -> List[str]:
    if "platinum_target" in row and row["platinum_target"]:
        tgt = row["platinum_target"]
        return [str(x) for x in (tgt if isinstance(tgt, list) else [tgt])]
    if subset in {"singleq","singleop","multiarith"} and row.get("output_answer"):
        return [str(row["output_answer"])]
    if subset == "gsm8k" and row.get("answer"):
        return [str(row["answer"])]
    if subset == "mmlu_math" and "answer" in row and "choices" in row:
        idx = int(row["answer"]); ch = row["choices"]
        if 0 <= idx < len(ch): return [str(ch[idx])]
    if subset in {"squad","hotpotqa"}:
        if row.get("answer"): return [str(row["answer"])]
        if isinstance(row.get("answers"), dict) and row["answers"].get("text"):
            return [str(row["answers"]["text"][0])]
    if subset == "winograd_wsc" and "label" in row and "options" in row:
        idx = int(row["label"]); ops = row["options"]
        if 0 <= idx < len(ops): return [str(ops[idx])]
    for k in ("target","label","output_answer","answer"):
        if k in row and row[k] not in (None,""): return [str(row[k])]
    return [""]  # last resort

def verdict(expected_vals: List[str], model_out: str) -> bool:
    # numeric path if both sides parse to numbers
    mn = extract_final_number(model_out)
    ens = [extract_final_number(e) for e in expected_vals]
    if mn is not None and any(e is not None for e in ens):
        return any(e is not None and abs(mn - e) <= 1e-6 for e in ens)
    # text path
    m_norm = normalize_text(model_out)
    return any((n := normalize_text(e)) == m_norm or n in m_norm or m_norm in n for e in expected_vals)

def paraphrase_prompt(p: str) -> str:
    # Light, deterministic tweak to keep semantics while changing surface form
    return p.rstrip() + "\nPlease answer concisely."

def init_gptcache(cache_obj: Cache, llm_string: str):
    # Semantic cache: ONNX embeddings + FAISS vector store + distance evaluator
    encoder = Onnx()  # uses GPTCache/paraphrase-albert-onnx
    cache_dir = f"cache_{_hash(llm_string)}"
    dm = manager_factory(
        "sqlite,faiss",
        data_dir=cache_dir,
        scalar_params={"sql_url": f"sqlite:///./{cache_dir}.db", "table_name": "ollama_cache"},
        vector_params={"dimension": encoder.dimension, "index_file_path": f"{cache_dir}.index"},
    )
    cfg = Config(similarity_threshold=SIM_THRESHOLD)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=encoder.to_embeddings,
        data_manager=dm,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=cfg,
    )

def make_log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME)
    return os.path.join(LOG_DIR, f"{ts}_{safe_model}_platinum_cache_demo.json")

# ----------------- main -----------------
def main():
    # Wire GPTCache (semantic) into LangChain globally
    set_llm_cache(LC_GPTCache(init_gptcache))

    llm = Ollama(base_url=BASE_URL, model=MODEL_NAME, temperature=0.0)
    results = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "base_url": BASE_URL,
        "subsets": SUBSETS,
        "index": INDEX,
        "cache_policy": "semantic (ONNX + FAISS)",
        "similarity_threshold": SIM_THRESHOLD,
        "items": []
    }

    print(f"Model: {MODEL_NAME}  |  Cache: semantic  |  Threshold: {SIM_THRESHOLD}")
    print(f"Subsets: {', '.join(SUBSETS)}\n")

    for subset in SUBSETS:
        ds = load_dataset("madrylab/platinum-bench", subset, split="test")
        if "cleaning_status" in ds.column_names:
            ds = ds.filter(lambda x: x["cleaning_status"] != "rejected")

        idx = INDEX if 0 <= INDEX < len(ds) else 0
        row = ds[idx]
        q = pick_question(row)
        exp = expected_candidates(row, subset)
        q2 = paraphrase_prompt(q)

        # First call (seed cache; likely MISS)
        t0 = time.time()
        out1 = llm.invoke(q).strip()
        dur1 = time.time() - t0

        # Second call (paraphrase; expect semantic HIT)
        t1 = time.time()
        out2 = llm.invoke(q2).strip()
        dur2 = time.time() - t1

        ok1 = verdict(exp, out1)
        ok2 = verdict(exp, out2)

        # crude "hit" heuristic: identical outputs and faster second call
        semantic_hit = (out2 == out1) and (dur2 < max(0.25, dur1 * 0.7))

        print(f"[{subset}] seed: {dur1:.2f}s  |  paraphrase: {dur2:.2f}s  |  "
              f"match1={'✓' if ok1 else '✗'}  match2={'✓' if ok2 else '✗'}  "
              f"semantic_hit={'✓' if semantic_hit else '·'}")

        results["items"].append({
            "subset": subset,
            "row_index": idx,
            "question": q,
            "paraphrase_question": q2,
            "expected": exp,
            "seed_output": out1,
            "paraphrase_output": out2,
            "seed_latency_sec": dur1,
            "paraphrase_latency_sec": dur2,
            "match_seed": bool(ok1),
            "match_paraphrase": bool(ok2),
            "semantic_hit_likely": bool(semantic_hit),
        })

    # write one JSON log
    log_path = make_log_path()
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nLog saved to: {log_path}")

if __name__ == "__main__":
    import datetime as dt
    main()

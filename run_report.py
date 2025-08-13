#!/usr/bin/env python3
"""
run_report.py
Generates a tidy report (Markdown + CSV) for:
- Ollama server & model status
- Smoke generate timing
- Platinum Bench dataset load
- GPTCache sanity with a dummy LLM
- Full pipeline (GPTCache + Ollama) on N prompts, 2 passes
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path

# GPTCache imports
from gptcache import Cache
from gptcache.adapter.api import get, put
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Datasets
from datasets import load_dataset

# ---------- Configuration ----------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
N = int(os.getenv("PB_N", "3"))  # number of prompts to test

# Output files
out_dir = Path(os.getenv("PB_OUT_DIR", "."))
out_dir.mkdir(parents=True, exist_ok=True)
md_path = out_dir / "gptcache_platinumbench_report.md"
csv_path = out_dir / "gptcache_platinumbench_runs.csv"

results_rows = []
report_lines = []

def section(title):
    report_lines.append(f"\\n## {title}\\n")

def line(s=""):
    report_lines.append(s)

def safe_json(obj):
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)

def main():
    # ---------- 1) Server & model checks ----------
    section("Ollama Server & Model Check")
    server_ok = False
    tags_json = None
    t0 = time.time()
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        tags_json = r.json()
        server_ok = True
        dur = time.time() - t0
        line(f"- ✅ **Server reachable**: `{OLLAMA_HOST}` in {dur:.2f}s")
        line(f"- Models: `{[m.get('name') for m in tags_json.get('models', [])]}`")
    except Exception as e:
        dur = time.time() - t0
        line(f"- ❌ **Server NOT reachable** after {dur:.2f}s")
        line(f"  - Error: `{repr(e)}`")

    results_rows.append({
        "step": "server_check",
        "status": "ok" if server_ok else "fail",
        "duration_s": round(dur, 3),
        "details": safe_json(tags_json) if tags_json else ""
    })

    # ---------- 2) Smoke generate ----------
    section("Ollama Generate Smoke Test")
    smoke_ok = False
    smoke_text = ""
    smoke_duration = None

    if server_ok:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Say hi in 5 words.",
            "stream": False,
            "options": {"num_predict": 32},
        }
        t0 = time.time()
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=300)
            r.raise_for_status()
            j = r.json()
            smoke_text = (j.get("response") or "").strip()
            smoke_ok = True
            smoke_duration = time.time() - t0
            line(f"- ✅ **Generate succeeded** with model `{OLLAMA_MODEL}` in {smoke_duration:.2f}s")
            line(f"- Output: `{smoke_text}`")
        except Exception as e:
            smoke_duration = time.time() - t0
            line(f"- ❌ **Generate failed** after {smoke_duration:.2f}s")
            line(f"  - Error: `{repr(e)}`")

    results_rows.append({
        "step": "smoke_generate",
        "status": "ok" if smoke_ok else "fail",
        "duration_s": round(smoke_duration or 0.0, 3),
        "details": smoke_text
    })

    # ---------- 3) Dataset check ----------
    section("Dataset Load Check")
    ds_ok = False
    bench_name = "gsm8k"
    num_prompts = 0
    first_question = ""

    t0 = time.time()
    try:
        ds = load_dataset("madrylab/platinum-bench", name=bench_name, split="test")
        num_prompts = len(ds)
        first_question = ds[0]["question"]
        ds_ok = True
        dur = time.time() - t0
        line(f"- ✅ **Loaded dataset** `madrylab/platinum-bench` (`{bench_name}`) with **{num_prompts}** items in {dur:.2f}s")
        line(f"- First question: `{first_question}`")
    except Exception as e:
        dur = time.time() - t0
        line(f"- ❌ **Dataset load failed** after {dur:.2f}s")
        line(f"  - Error: `{repr(e)}`")

    results_rows.append({
        "step": "dataset_load",
        "status": "ok" if ds_ok else "fail",
        "duration_s": round(dur, 3),
        "details": f"items={num_prompts}"
    })

    # ---------- 4) Cache sanity with dummy LLM ----------
    section("GPTCache Sanity (Dummy LLM)")
    cache_dummy_ok = False
    dummy_details = ""

    try:
        embedding_model = Onnx()
        chat_cache = Cache()
        chat_cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=embedding_model.to_embeddings,
            data_manager=get_data_manager(
                CacheBase("sqlite", path="cache_report.db"),
                VectorBase("faiss", dimension=embedding_model.dimension),
            ),
            similarity_evaluation=SearchDistanceEvaluation(),
        )

        def dummy_llm(prompt: str) -> str:
            return f"Dummy response for: {prompt[::-1]}"

        prompts = [
            "What is GPTCache?",
            "Tell me about GPTCache.",
            "What is GPTCache?"
        ]
        hits = 0
        misses = 0
        for p in prompts:
            d = {"prompt": p}
            ans = get(p, cache_obj=chat_cache, data=d)
            if ans is None:
                misses += 1
                ans = dummy_llm(p)
                put(d, ans, cache_obj=chat_cache)  # NOTE: data first (GPTCache 0.1.44)
            else:
                hits += 1

        cache_dummy_ok = (hits >= 1 and misses >= 2)
        line(f"- ✅ Hits: **{hits}**, Misses: **{misses}**")
        dummy_details = f"hits={hits},misses={misses}"
    except Exception as e:
        line(f"- ❌ Cache sanity failed: `{repr(e)}`")
        dummy_details = repr(e)

    results_rows.append({
        "step": "cache_dummy_sanity",
        "status": "ok" if cache_dummy_ok else "fail",
        "duration_s": None,
        "details": dummy_details
    })

    # ---------- 5) Full pipeline ----------
    section("Full Pipeline: GPTCache + Ollama (N items, 2 passes)")
    pipeline_ok = False
    full_details = ""

    if server_ok and ds_ok:
        try:
            # fresh cache for the pipeline to measure hits clearly
            embedding_model = Onnx()
            bench_cache = Cache()
            bench_cache.init(
                pre_embedding_func=get_prompt,
                embedding_func=embedding_model.to_embeddings,
                data_manager=get_data_manager(
                    CacheBase("sqlite", path="cache_bench.db"),
                    VectorBase("faiss", dimension=embedding_model.dimension),
                ),
                similarity_evaluation=SearchDistanceEvaluation(),
            )

            def ollama_generate(prompt: str, num_predict: int = 64, timeout_s: int = 300) -> str:
                r = requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": num_predict}},
                    timeout=timeout_s,
                )
                r.raise_for_status()
                return r.json()["response"].strip()

            # Pass 1
            pass1_calls = 0
            t0 = time.time()
            for i in range(min(N, num_prompts)):
                prompt = ds[i]["question"].strip()
                d = {"prompt": prompt}
                out = get(prompt, cache_obj=bench_cache, data=d)
                if out is None:
                    out = ollama_generate(prompt, num_predict=64, timeout_s=300)
                    put(d, out, cache_obj=bench_cache)
                    pass1_calls += 1
            t_pass1 = time.time() - t0

            # Pass 2
            pass2_calls = 0
            t0 = time.time()
            for i in range(min(N, num_prompts)):
                prompt = ds[i]["question"].strip()
                d = {"prompt": prompt}
                out = get(prompt, cache_obj=bench_cache, data=d)
                if out is None:
                    out = ollama_generate(prompt, num_predict=64, timeout_s=300)
                    put(d, out, cache_obj=bench_cache)
                    pass2_calls += 1
            t_pass2 = time.time() - t0

            pipeline_ok = (pass1_calls == min(N, num_prompts) and pass2_calls == 0)
            line(f"- Pass 1: LLM calls **{pass1_calls}** in {t_pass1:.2f}s")
            line(f"- Pass 2: LLM calls **{pass2_calls}** in {t_pass2:.2f}s (expect 0)")

            full_details = f"pass1_calls={pass1_calls},pass2_calls={pass2_calls},t1={t_pass1:.2f},t2={t_pass2:.2f}"
        except Exception as e:
            line(f"- ❌ Pipeline error: `{repr(e)}`")
            full_details = repr(e)
    else:
        line("- Skipped (server or dataset not ready)")

    results_rows.append({
        "step": "full_pipeline",
        "status": "ok" if pipeline_ok else "fail",
        "duration_s": None,
        "details": full_details
    })

    # ---------- Write outputs ----------
    section("Summary Table")
    df = pd.DataFrame(results_rows, columns=["step", "status", "duration_s", "details"])
    df["duration_s"] = df["duration_s"].apply(lambda x: "" if x is None else x)

    # Markdown table
    line("\\n| step | status | duration_s | details |")
    line("|---|---|---:|---|")
    for _, row in df.iterrows():
        details = str(row["details"]).replace("\\n", " ")
        line(f"| {row['step']} | {row['status']} | {row['duration_s']} | {details} |")

    md_path.write_text("# GPTCache + Ollama + Platinum Bench — Quick Report\\n" + "\\n".join(report_lines), encoding="utf-8")
    df.to_csv(csv_path, index=False)

    print(f"Report saved to: {md_path}")
    print(f"CSV saved to:    {csv_path}")

if __name__ == "__main__":
    main()

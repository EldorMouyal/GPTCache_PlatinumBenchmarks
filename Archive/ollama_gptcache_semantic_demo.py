#!/usr/bin/env python3
"""
Semantic (vector) GPTCache with Ollama (LangChain).

Flow:
  1) MISS on base prompt
  2) HIT on a paraphrase (vector similarity)
  3) MISS on unrelated prompt

Vector store:
  - Default: FAISS (Windows-friendly)
  - Optional: HNSWlib (set env GPTCACHE_VSTORE=hnswlib)

Install:
  pip install -U langchain langchain-community gptcache onnxruntime transformers
  pip install -U faiss-cpu
  # (optional) pip install -U hnswlib
  # (recommended) pip install -U langchain-ollama

Run:
  ollama serve
  ollama pull llama3:8b
  python ollama_gptcache_semantic_demo_v2.py
"""

import os
import time
import hashlib
import logging

# Prefer the new langchain-ollama package; fall back if not installed
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama  # deprecated, but works

from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache as LC_GPTCache

# GPTCache pieces
from gptcache import Cache, Config
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import manager_factory
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# ---------------- Config ----------------
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

BASE_PROMPT = "In one concise sentence, define GPTCache and mention one benefit."
PARAPHRASE_PROMPT = "Briefly describe what GPTCache is and include one advantage."
DIFFERENT_PROMPT = "What is the capital of France?"

SIM_THRESHOLD = float(os.getenv("GPTCACHE_SIM_THRESHOLD", "0.75"))
VSTORE = os.getenv("GPTCACHE_VSTORE", "faiss").lower()  # 'faiss' (default) or 'hnswlib'


def _hash(s: str) -> str:
    import hashlib as _h
    return _h.sha256(s.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm_string: str):
    """
    Initialize GPTCache for semantic similarity with ONNX embeddings,
    using FAISS (default) or HNSWlib (if GPTCACHE_VSTORE=hnswlib).
    """
    encoder = Onnx()  # GPTCache/paraphrase-albert-onnx, exposes .dimension

    cache_dir = f"semantic_cache_{_hash(llm_string)}"
    sql_url = f"sqlite:///./{cache_dir}.db"

    if VSTORE == "hnswlib":
        # Tuned params so first searches don't explode on Windows
        dm = manager_factory(
            "sqlite,hnswlib",
            data_dir=cache_dir,
            scalar_params={"sql_url": sql_url, "table_name": "ollama_cache"},
            vector_params={
                "dimension": encoder.dimension,
                # HNSW tuning: larger graph / ef helps avoid early knn_query issues
                "M": 48,                      # graph degree
                "ef_construction": 200,       # build-time accuracy
                "ef": 200,                    # query-time accuracy (>= top_k)
                "space": "cosine",            # cosine distance for sentence embeddings
                "max_elements": 10000,        # capacity
            },
        )
    else:
        # FAISS (recommended default)
        dm = manager_factory(
            "sqlite,faiss",
            data_dir=cache_dir,
            scalar_params={"sql_url": sql_url, "table_name": "ollama_cache"},
            vector_params={
                "dimension": encoder.dimension,
                "index_file_path": f"{cache_dir}.index",  # persist index to disk
            },
        )

    cfg = Config(similarity_threshold=SIM_THRESHOLD)

    cache_obj.init(
        pre_embedding_func=get_prompt,            # use the raw prompt text
        embedding_func=encoder.to_embeddings,     # text -> vector
        data_manager=dm,
        similarity_evaluation=SearchDistanceEvaluation(),  # distance-based match
        config=cfg,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Wire GPTCache into LangChain
    set_llm_cache(LC_GPTCache(init_gptcache))

    # Ollama LLM
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE)

    print(f"Using Ollama model: {MODEL_NAME} @ {OLLAMA_BASE_URL}")
    print(f"Vector store: {VSTORE}  |  Similarity threshold: {SIM_THRESHOLD}\n")

    # 1) MISS (seed)
    t0 = time.time()
    out1 = llm.invoke(BASE_PROMPT)
    dur1 = time.time() - t0
    print("1) First call (expected MISS):")
    print("   Prompt:", BASE_PROMPT)
    print("   Output:", out1.strip()[:220], "\n")
    print(f"   Duration: {dur1:.3f}s\n")

    # 2) HIT (semantic paraphrase)
    t1 = time.time()
    out2 = llm.invoke(PARAPHRASE_PROMPT)
    dur2 = time.time() - t1
    print("2) Paraphrase call (expected semantic HIT):")
    print("   Prompt:", PARAPHRASE_PROMPT)
    print("   Output:", out2.strip()[:220], "\n")
    print(f"   Duration: {dur2:.3f}s")

    if out2 == out1:
        print("   ✅ Semantic cache HIT confirmed (identical outputs).")
    else:
        print("   ℹ️ If this wasn't a HIT, try lowering GPTCACHE_SIM_THRESHOLD (e.g., 0.70).")
    print()

    # 3) MISS (unrelated)
    t2 = time.time()
    out3 = llm.invoke(DIFFERENT_PROMPT)
    dur3 = time.time() - t2
    print("3) Different question (expected MISS):")
    print("   Prompt:", DIFFERENT_PROMPT)
    print("   Output:", out3.strip()[:220], "\n")
    print(f"   Duration: {dur3:.3f}s\n")

    # Timing sanity: paraphrase likely faster than first call
    clearly_faster = dur2 < max(0.25, dur1 * 0.6)
    if out2 == out1 and clearly_faster:
        print("🎯 Looks solid: paraphrase was clearly faster & identical => semantic HIT.")
    else:
        print("⚠️ If HIT not obvious, lower threshold or use a slower LLM for clearer diffs.")


if __name__ == "__main__":
    main()

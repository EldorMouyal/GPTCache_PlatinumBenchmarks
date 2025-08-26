#!/usr/bin/env python3
"""
Exact-match GPTCache with Ollama (via LangChain), no FAISS required.

- First call: MISS (hits the LLM)
- Second call (same prompt): HIT (served from GPTCache)
"""

import os
import time
import hashlib

# LangChain + Ollama
from langchain_community.llms import Ollama
from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache as LC_GPTCache

# GPTCache bits
import gptcache
from gptcache import Cache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import manager_factory
from gptcache.similarity_evaluation import ExactMatchEvaluation  # exact-match evaluator

# ---------- Config ----------
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PROMPT = "In one sentence: What is GPTCache and why is it useful?"
TEMPERATURE = 0.0  # deterministic output for equality checks


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm_string: str):
    """
    Initialize GPTCache for exact-match using the in-memory 'map' data manager.
    A per-LLM cache directory is used to avoid collisions across models.
    """
    cache_dir = f"map_cache_{_hash(llm_string)}"
    cache_obj.init(
        pre_embedding_func=get_prompt,                         # key on raw prompt text
        data_manager=manager_factory(manager="map", data_dir=cache_dir),
        similarity_evaluation=ExactMatchEvaluation(),          # strict exact-match policy
    )


def main():
    # Wire up LangChain to use GPTCache
    set_llm_cache(LC_GPTCache(init_gptcache))

    # Create Ollama LLM
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    print(f"Using Ollama model: {MODEL_NAME} @ {OLLAMA_BASE_URL}")
    print(f"Prompt: {PROMPT}\n")

    # 1) MISS
    t0 = time.time()
    out1 = llm.invoke(PROMPT)
    dur1 = time.time() - t0
    print("First call (expected MISS):")
    print(out1.strip()[:200], "\n")
    print(f"Duration: {dur1:.3f}s\n")

    # 2) HIT
    t1 = time.time()
    out2 = llm.invoke(PROMPT)
    dur2 = time.time() - t1
    print("Second call (expected HIT):")
    print(out2.strip()[:200], "\n")
    print(f"Duration: {dur2:.3f}s\n")

    identical = (out1 == out2)
    clearly_faster = dur2 < max(0.25, dur1 * 0.5)
    if identical and clearly_faster:
        print("✅ Cache HIT verified: identical outputs and clearly faster second call.")
    elif identical:
        print("✅ Cache likely HIT: identical outputs (timing not clearly faster).")
    else:
        print("⚠️ Outputs differ — check temperature/model settings.")


if __name__ == "__main__":
    main()

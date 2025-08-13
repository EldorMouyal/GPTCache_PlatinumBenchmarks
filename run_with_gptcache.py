import os
import requests
from datasets import load_dataset

from gptcache import Cache
from gptcache.adapter.api import get, put
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


# inside run_with_gptcache.py
from ollama_client import generate as ollama_generate



# ---------- GPTCache (explicit instance) ----------
embedding_model = Onnx()
chat_cache = Cache()
chat_cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=embedding_model.to_embeddings,
    data_manager=get_data_manager(
        CacheBase("sqlite", path="cache.db"),
        VectorBase("faiss", dimension=embedding_model.dimension),
    ),
    similarity_evaluation=SearchDistanceEvaluation(),
)

# ---------- Ollama endpoint (inside the container) ----------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")  # ensure you pulled this

def ollama_generate(prompt: str, timeout_s: int = 240) -> str:
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout_s,  # allow long first-call warmup
    )
    r.raise_for_status()
    return r.json()["response"].strip()

# ---------- Dataset ----------
BENCH_NAME = "gsm8k"
dataset = load_dataset("madrylab/platinum-bench", name=BENCH_NAME, split="test")
print(f"Loaded {len(dataset)} prompts from Platinum Benchmark ({BENCH_NAME})")

# ---------- Run a small sample twice to check cache ----------
N = 3
def run_pass(label: str, count: int):
    print(f"\n=== {label} ===")
    llm_calls = 0
    for i in range(count):
        prompt = dataset[i]["question"].strip()
        data = {"prompt": prompt}  # match get_prompt preprocessor

        out = get(prompt, cache_obj=chat_cache, data=data)
        hit = out is not None
        if not hit:
            # MISS -> call LLM -> store in cache
            out = ollama_generate(prompt, num_predict=64, timeout_s=300, tries=2)
            # NOTE: 'data' goes as the first positional arg to put(...)
            put(data, out, cache_obj=chat_cache)
            llm_calls += 1

        print(f"\n🧠 Prompt {i+1}: {prompt}\n"
              f"{'🟢 Cached Response' if hit else '🤖 LLM Response'}: {out[:240]}...")

    print(f"\nLLM calls in this pass: {llm_calls}")
    return llm_calls

first_llm_calls = run_pass("First pass (expect LLM calls)", N)
second_llm_calls = run_pass("Second pass (expect cache hits)", N)

print("\n--- Summary ---")
print(f"LLM calls in first pass:  {first_llm_calls}")
print(f"LLM calls in second pass: {second_llm_calls}  (should be 0 if cached)")

import os
import requests
from datasets import load_dataset
from gptcache import cache
from gptcache.adapter import get_data, Cache

# GPTCache setup (default config)
cache.init()

# Setup Ollama model name
OLLAMA_MODEL = "gemma3"
@Cache
def ollama_generate(prompt: str) -> str:
    response = requests.post(
        "http://host.docker.internal:11434/api/generate", # works from Docker
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# Benchmark dataset
BENCH_NAME = "gsm8k"
dataset = load_dataset("madrylab/platinum-bench", name=BENCH_NAME, split="test")
print(f"Loaded {len(dataset)} prompts from Platinum Benchmark ({BENCH_NAME})")

# Run 3 prompts through GPTCache + Ollama
for i in range(3):
    prompt = dataset[i]["question"].strip()

    print(f"\n🧠 Prompt {i+1}: {prompt}")
    reply = ollama_generate(prompt)
    print(f"🤖 Ollama Response (via GPTCache): {reply}")

dm = get_data()
print(f"\nCache entries: {len(dm)}")

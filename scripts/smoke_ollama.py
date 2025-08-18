import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models.ollama import setup_model, generate

cfg = {
    "provider": "ollama",
    "name": "gemma2:9b",  # Ensure you've pulled this locally on your machine
    "params": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 64},
    # "base_url": "http://localhost:11434",  # optional override
    # "timeout_sec": 120,
}
setup_model(cfg)

print(">> Querying gemma2:9b via Ollama...")
out = generate("In one sentence, what is caching for LLMs?")
print("=== MODEL OUTPUT ===")
print(out)

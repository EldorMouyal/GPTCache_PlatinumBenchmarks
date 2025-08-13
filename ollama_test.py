# File: ollama_test.py
import os
import requests
# File: ollama_test.py
from ollama_client import generate

resp = generate("Say hi in 5 words.", num_predict=32, timeout_s=300, tries=2)
print("Response:", resp)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

prompt = "What is gravity in simple terms?"
r = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    timeout=60,
)
r.raise_for_status()
print("Response:", r.json()["response"].strip())

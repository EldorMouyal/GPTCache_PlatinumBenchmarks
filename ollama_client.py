# File: ollama_client.py
import os
import time
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

def generate(prompt: str,
             num_predict: int = 64,
             timeout_s: int = 300,
             tries: int = 2) -> str:
    """Call Ollama with small output, longer timeout, and 1 retry for cold starts."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict
        }
    }
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate",
                              json=payload,
                              timeout=timeout_s)
            r.raise_for_status()
            return r.json()["response"].strip()
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            last_err = e
            # brief backoff; first call often warms the model
            time.sleep(5 * attempt)
    # if still failing, raise the last error
    raise last_err

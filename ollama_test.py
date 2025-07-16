import requests

OLLAMA_MODEL = "gemma3"

prompt = "What is gravity in simple terms?"
response = requests.post(
    "http://host.docker.internal:11434/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
)
response.raise_for_status()
print("Response:", response.json()["response"].strip())

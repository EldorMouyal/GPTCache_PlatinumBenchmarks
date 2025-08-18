#!/usr/bin/env python3
"""
Real Ollama adapter (Step 4): calls the local Ollama server via HTTP.

Requirements:
- Ollama running locally (default host: http://localhost:11434)
- Model pulled, e.g.:  `ollama pull gemma2:9b`

Public API expected by runner:
    setup_model(config: dict) -> None
    generate(prompt: str, params: dict | None = None) -> str
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import json
import urllib.request
import urllib.error
import socket

_OLLAMA_URL = "http://localhost:11434"
_MODEL_NAME = "gemma2:9b"
_DEFAULT_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 256,
}
_TIMEOUT = 120  # seconds

def _http_post(path: str, payload: Dict[str, Any], timeout: int = _TIMEOUT) -> Dict[str, Any]:
    url = f"{_OLLAMA_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Ollama's /api/generate can stream; if we set stream:false, it's a single JSON
            text = resp.read().decode("utf-8")
            # Some Ollama versions may send multiple JSON lines if stream flag ignored;
            # take the last complete JSON object with a "response" or "message".
            # Try parse as one JSON; if fails, split by lines and parse last non-empty.
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                last = None
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        last = obj
                    except json.JSONDecodeError:
                        continue
                if last is None:
                    raise
                return last
    except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout) as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e


def setup_model(config: Dict[str, Any]) -> None:
    """Configure target model and default params from experiment config."""
    global _MODEL_NAME, _DEFAULT_PARAMS, _OLLAMA_URL, _TIMEOUT
    model = (config or {}).get("name") or _MODEL_NAME
    _MODEL_NAME = str(model)

    params = (config or {}).get("params") or {}
    # Only keep known numeric params; ignore unknown keys to remain import-safe
    merged = dict(_DEFAULT_PARAMS)
    for key in ("temperature", "top_p", "max_tokens"):
        if key in params:
            merged[key] = params[key]
    _DEFAULT_PARAMS = merged

    # Optional advanced knobs (not required by runner)
    base_url = (config or {}).get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        _OLLAMA_URL = base_url.rstrip("/")

    timeout = (config or {}).get("timeout_sec")
    if isinstance(timeout, (int, float)) and timeout > 0:
        _TIMEOUT = int(timeout)


def generate(prompt: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Call Ollama /api/generate and return the final text response."""
    body = {
        "model": _MODEL_NAME,
        "prompt": str(prompt),
        # force non-stream to simplify; some versions still line-stream—handled above
        "stream": False,
    }

    # Map our generic params into Ollama's expected keys
    eff = dict(_DEFAULT_PARAMS)
    if params:
        for k in ("temperature", "top_p", "max_tokens"):
            if k in params:
                eff[k] = params[k]

    # Ollama uses `options` for sampling; truncate/max_tokens may be model-specific
    body["options"] = {
        "temperature": float(eff["temperature"]),
        "top_p": float(eff["top_p"]),
    }
    # For gemma2 family, truncation is typically `num_predict` (tokens to generate)
    body["num_predict"] = int(eff["max_tokens"])

    res = _http_post("/api/generate", body, timeout=_TIMEOUT)

    # The unified (non-stream) response typically has:
    # { "model":..., "created_at":..., "response": "...", "done": true, ... }
    # If streaming was returned line-wise, we parsed the last object above.
    if "response" in res and isinstance(res["response"], str):
        return res["response"]

    # Fallback: some endpoints return {"message":{"content": "..."}} (chat API shape)
    msg = res.get("message", {})
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]

    # If nothing matches, surface the raw payload for debugging
    return json.dumps(res, ensure_ascii=False)

# src/models/ollama.py

from __future__ import annotations
from typing import Any, Dict, Optional


def _import_ollama_class():
    """
    Prefer the first-party langchain_ollama provider. Fall back to the
    community provider if it's not installed.
    """
    try:
        from langchain_ollama import OllamaLLM as Ollama  # type: ignore
        return Ollama
    except Exception:
        from langchain_community.llms import Ollama  # type: ignore
        return Ollama


def build_llm(model: str, base_url: str, params: Optional[Dict[str, Any]] = None):
    """
    Build an Ollama-based LangChain LLM that supports .invoke(prompt).

    Args:
        model: The model name/tag available in your local Ollama (e.g., "llama3.1:8b").
        base_url: Ollama server URL (e.g., "http://localhost:11434").
        params: Extra generation parameters (taken directly from experiment.yaml).

    Returns:
        An instantiated LLM object (Runnable) with .invoke(prompt: str) -> str
    """
    import os
    
    Ollama = _import_ollama_class()
    kwargs: Dict[str, Any] = dict(params or {})

    # Allow environment variable to override config file base_url
    # Useful for Docker deployments and remote GPU servers
    effective_base_url = os.getenv("OLLAMA_BASE_URL", base_url)

    # Do NOT redefine temperature or any other keys here.
    # If experiment.yaml leaves it out, the provider's own defaults apply.
    return Ollama(base_url=effective_base_url, model=model, **kwargs)

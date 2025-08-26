# src/cache_strategies/none.py

"""
Cache Strategy: None

This module disables caching entirely.
It sets the LangChain LLM cache to None, ensuring that
every request goes directly to the model backend.

References:
- Detailed structure guide: PROJECT_STRUCTURE_GUIDE_DETAILED.md
- Demo alignment: matches behavior when GPTCache is not initialized.
"""

from langchain_core.globals import set_llm_cache


def setup_cache(cfg: dict) -> None:
    """
    Disable caching for the run.

    Args:
        cfg (dict): Experiment configuration dictionary (unused here).
    """
    set_llm_cache(None)

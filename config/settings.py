"""
config/settings.py
------------------
Centralised, validated application configuration.

All tuneable parameters live here and are sourced from environment
variables with sensible defaults.  Importing this module from any agent
or utility guarantees a single source of truth.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.warning("Config: %s=%r is not a valid integer; using default %d", key, raw, default)
        return default


def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.warning("Config: %s=%r is not a valid float; using default %.2f", key, raw, default)
        return default


# LLM settings
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = _get_env_float("GROQ_TEMPERATURE", 0.1)
GROQ_MAX_TOKENS: int = _get_env_int("GROQ_MAX_TOKENS", 4096)
GROQ_MAX_RETRIES: int = _get_env_int("GROQ_MAX_RETRIES", 3)
GROQ_TIMEOUT: float = _get_env_float("GROQ_TIMEOUT", 60.0)

# Workflow settings
MAX_ITERATIONS: int = _get_env_int("MAX_ITERATIONS", 6)
MAX_TOOL_ITERATIONS: int = _get_env_int("MAX_TOOL_ITERATIONS", 4)

# Search settings
SEARCH_MAX_RESULTS: int = _get_env_int("SEARCH_MAX_RESULTS", 6)
NEWS_MAX_RESULTS: int = _get_env_int("NEWS_MAX_RESULTS", 5)

# Output settings
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "research_outputs"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def validate() -> None:
    """
    Assert that required configuration is present.

    Raises:
        OSError: If GROQ_API_KEY is missing.
    """
    if not GROQ_API_KEY:
        raise OSError(
            "GROQ_API_KEY is not set.  "
            "Copy .env.example to .env and add your key from https://console.groq.com/keys"
        )

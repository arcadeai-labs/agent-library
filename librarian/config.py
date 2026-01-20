"""
Configuration management for librarian.

This module provides centralized configuration with environment variable overrides
and sensible defaults for all system parameters.
"""

import os
from pathlib import Path


def safe_int(value: str | None, default: int) -> int:
    """
    Safely convert a value to int, returning default on failure.

    Args:
        value: The value to convert.
        default: Default value if conversion fails.

    Returns:
        The converted integer or the default value.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: str | None, default: float) -> float:
    """
    Safely convert a value to float, returning default on failure.

    Args:
        value: The value to convert.
        default: Default value if conversion fails.

    Returns:
        The converted float or the default value.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# Path Configuration
# =============================================================================

DOCUMENTS_PATH = os.path.abspath(os.path.expanduser(os.getenv("DOCUMENTS_PATH", "./documents")))

DATABASE_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("DATABASE_PATH", "~/.librarian/index.db"))
)

# =============================================================================
# Embedding Configuration
# =============================================================================

# Provider selection: "local" (sentence-transformers) or "openai" (OpenAI-compatible API)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# Local embedding settings (sentence-transformers)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = safe_int(os.getenv("EMBEDDING_DIMENSION"), 384)

# OpenAI-compatible API settings (default: local qwen3-embedding service)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://mansamura:7171/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "qwen3-embedding-06b")
OPENAI_EMBEDDING_DIMENSION = safe_int(os.getenv("OPENAI_EMBEDDING_DIMENSION"), 1024)
OPENAI_EMBEDDING_BATCH_SIZE = safe_int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE"), 64)

# Instruction-based embedding settings (for Qwen3-Embedding and similar models)
# Task description used with GoEmbed instruction config (type: "query", task: "...")
EMBEDDING_QUERY_INSTRUCTION = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION",
    "Given a step of a task in the form of a query, return information from "
    "documents that is relevant to addressing the query.",
)

# =============================================================================
# Chunking Configuration
# ====================================e=========================================

CHUNK_SIZE = safe_int(os.getenv("CHUNK_SIZE"), 512)
CHUNK_OVERLAP = safe_int(os.getenv("CHUNK_OVERLAP"), 50)
MIN_CHUNK_SIZE = safe_int(os.getenv("MIN_CHUNK_SIZE"), 50)

# =============================================================================
# Search Configuration
# =============================================================================

SEARCH_LIMIT = safe_int(os.getenv("SEARCH_LIMIT"), 10)
MMR_LAMBDA = safe_float(os.getenv("MMR_LAMBDA"), 0.1)  # 0.5 for more diverse results
HYBRID_ALPHA = safe_float(os.getenv("HYBRID_ALPHA"), 0.7)

# =============================================================================
# Background Processing Configuration
# =============================================================================

INDEX_POLL_INTERVAL = safe_float(os.getenv("INDEX_POLL_INTERVAL"), 60.0)
INDEX_START_DELAY = safe_float(os.getenv("INDEX_START_DELAY"), 5.0)


def ensure_directories() -> None:
    """Ensure required directories exist (database only, not documents)."""
    # Only create database directory - documents are managed via 'libr sources add'
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)

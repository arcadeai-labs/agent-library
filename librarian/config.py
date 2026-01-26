"""
Configuration management for librarian multi-modal support.

Centralized configuration with environment variable overrides and
sensible defaults for all system parameters.
"""

import os
from pathlib import Path

from librarian.types import AssetType, ChunkingStrategy, EmbeddingModality


def safe_int(value: str | None, default: int) -> int:
    """Convert value to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: str | None, default: float) -> float:
    """Convert value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: str | None, default: bool) -> bool:
    """Convert value to bool, returning default on failure."""
    if value is None:
        return default
    return str(value).lower() in ("true", "1", "yes", "on")


# =============================================================================
# Path Configuration
# =============================================================================

DOCUMENTS_PATH = os.path.abspath(os.path.expanduser(os.getenv("DOCUMENTS_PATH", "./documents")))

DATABASE_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("DATABASE_PATH", "~/.librarian/index.db"))
)

SOURCES_CONFIG_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("SOURCES_CONFIG_PATH", "~/.librarian/sources.json"))
)

# =============================================================================
# Text Embedding Configuration
# =============================================================================

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = safe_int(os.getenv("EMBEDDING_DIMENSION"), 384)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:7171/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "qwen3-embedding-06b")
OPENAI_EMBEDDING_DIMENSION = safe_int(os.getenv("OPENAI_EMBEDDING_DIMENSION"), 1024)
OPENAI_EMBEDDING_BATCH_SIZE = safe_int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE"), 64)

EMBEDDING_QUERY_INSTRUCTION = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION",
    "Given a query, return relevant information from documents.",
)

# =============================================================================
# Code Embedding Configuration
# =============================================================================

ENABLE_CODE_EMBEDDINGS = safe_bool(os.getenv("ENABLE_CODE_EMBEDDINGS"), False)
CODE_EMBEDDING_MODEL = os.getenv("CODE_EMBEDDING_MODEL", "microsoft/codebert-base")
CODE_EMBEDDING_DIMENSION = safe_int(os.getenv("CODE_EMBEDDING_DIMENSION"), 768)
CODE_EMBEDDING_PROVIDER = os.getenv("CODE_EMBEDDING_PROVIDER", "local")

# =============================================================================
# Vision Embedding Configuration
# =============================================================================

ENABLE_VISION_EMBEDDINGS = safe_bool(os.getenv("ENABLE_VISION_EMBEDDINGS"), False)
VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
VISION_EMBEDDING_DIMENSION = safe_int(os.getenv("VISION_EMBEDDING_DIMENSION"), 512)

# =============================================================================
# OCR Configuration
# =============================================================================

ENABLE_OCR = safe_bool(os.getenv("ENABLE_OCR"), True)
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_CONFIG = os.getenv("OCR_CONFIG", "--psm 3")  # Page segmentation mode
OCR_MIN_CONFIDENCE = safe_int(os.getenv("OCR_MIN_CONFIDENCE"), 0)  # 0-100, 0 = no filtering

# =============================================================================
# Asset Type Configuration
# =============================================================================

DEFAULT_ASSET_TYPES_STR = os.getenv("DEFAULT_ASSET_TYPES", "text,code")
DEFAULT_ASSET_TYPES = [
    AssetType(t.strip()) for t in DEFAULT_ASSET_TYPES_STR.split(",") if t.strip()
]

# =============================================================================
# Chunking Configuration
# =============================================================================

CHUNK_SIZE = safe_int(os.getenv("CHUNK_SIZE"), 512)
CHUNK_OVERLAP = safe_int(os.getenv("CHUNK_OVERLAP"), 50)
MIN_CHUNK_SIZE = safe_int(os.getenv("MIN_CHUNK_SIZE"), 50)

# Code-specific chunking
CODE_CHUNK_STRATEGY_STR = os.getenv("CODE_CHUNK_STRATEGY", "code_blocks")
CODE_CHUNK_STRATEGY = ChunkingStrategy(CODE_CHUNK_STRATEGY_STR)
CODE_INCLUDE_CONTEXT = safe_bool(os.getenv("CODE_INCLUDE_CONTEXT"), True)
CODE_CONTEXT_LINES = safe_int(os.getenv("CODE_CONTEXT_LINES"), 5)

# PDF-specific chunking
PDF_CHUNK_STRATEGY_STR = os.getenv("PDF_CHUNK_STRATEGY", "pages")
PDF_CHUNK_STRATEGY = ChunkingStrategy(PDF_CHUNK_STRATEGY_STR)

# =============================================================================
# Search Configuration
# =============================================================================

SEARCH_LIMIT = safe_int(os.getenv("SEARCH_LIMIT"), 10)
MMR_LAMBDA = safe_float(os.getenv("MMR_LAMBDA"), 0.1)
HYBRID_ALPHA = safe_float(os.getenv("HYBRID_ALPHA"), 0.7)

# Cross-modal search
ENABLE_CROSS_MODAL_SEARCH = safe_bool(os.getenv("ENABLE_CROSS_MODAL_SEARCH"), True)
CROSS_MODAL_SIMILARITY_THRESHOLD = safe_float(os.getenv("CROSS_MODAL_SIMILARITY_THRESHOLD"), 0.7)

# Modality weights
MODALITY_WEIGHT_TEXT = safe_float(os.getenv("MODALITY_WEIGHT_TEXT"), 1.0)
MODALITY_WEIGHT_CODE = safe_float(os.getenv("MODALITY_WEIGHT_CODE"), 0.9)
MODALITY_WEIGHT_VISION = safe_float(os.getenv("MODALITY_WEIGHT_VISION"), 0.8)

MODALITY_WEIGHTS = {
    EmbeddingModality.TEXT: MODALITY_WEIGHT_TEXT,
    EmbeddingModality.CODE: MODALITY_WEIGHT_CODE,
    EmbeddingModality.VISION: MODALITY_WEIGHT_VISION,
}

# =============================================================================
# Codebase Management Configuration
# =============================================================================

CODEBASE_AUTO_DETECT = safe_bool(os.getenv("CODEBASE_AUTO_DETECT"), True)
CODEBASE_INDEX_TESTS = safe_bool(os.getenv("CODEBASE_INDEX_TESTS"), True)
CODEBASE_MAX_FILE_SIZE_KB = safe_int(os.getenv("CODEBASE_MAX_FILE_SIZE_KB"), 500)

# =============================================================================
# PDF Processing Configuration
# =============================================================================

ENABLE_PDF_PROCESSING = safe_bool(os.getenv("ENABLE_PDF_PROCESSING"), True)
PDF_OCR_ENABLED = safe_bool(os.getenv("PDF_OCR_ENABLED"), False)

# =============================================================================
# Image Processing Configuration
# =============================================================================

IMAGE_GENERATE_CAPTIONS = safe_bool(os.getenv("IMAGE_GENERATE_CAPTIONS"), False)
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "blip-base")

# =============================================================================
# Tool Behavior Configuration
# =============================================================================

TOOL_SEARCH_DEFAULT_LIMIT = safe_int(os.getenv("TOOL_SEARCH_DEFAULT_LIMIT"), 10)
TOOL_MAX_CONTEXT_LINES = safe_int(os.getenv("TOOL_MAX_CONTEXT_LINES"), 10)
CODE_MAX_DEPENDENCY_DEPTH = safe_int(os.getenv("CODE_MAX_DEPENDENCY_DEPTH"), 3)
CODE_MAX_REFERENCES = safe_int(os.getenv("CODE_MAX_REFERENCES"), 50)

# =============================================================================
# Background Processing Configuration
# =============================================================================

INDEX_POLL_INTERVAL = safe_float(os.getenv("INDEX_POLL_INTERVAL"), 60.0)
INDEX_START_DELAY = safe_float(os.getenv("INDEX_START_DELAY"), 5.0)


def ensure_directories() -> None:
    """Ensure required directories exist."""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SOURCES_CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)

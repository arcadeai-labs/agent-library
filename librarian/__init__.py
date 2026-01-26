"""
Librarian - Context Management Service.

A complete system for maintaining, indexing, ingesting, and retrieving
documents through the Model Context Protocol (MCP).

Features:
- SQLite-based storage with vector search (sqlite-vec)
- Full-text search (FTS5) with BM25 ranking
- Hybrid search combining vector and keyword search
- Max Marginal Relevance (MMR) for diverse results
- Configurable embedding providers (local or OpenAI)
- Intelligent text chunking with overlap
- Time-bounded search with timeframe filters

Usage:
    # For MCP server
    from librarian.server import app

    # For processing
    from librarian.processing import ProcessingManager

    # For CLI
    librarian --help
"""

from librarian.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATABASE_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HYBRID_ALPHA,
    MMR_LAMBDA,
    SEARCH_LIMIT,
)

__version__ = "0.8.0"

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DATABASE_PATH",
    "DOCUMENTS_PATH",
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER",
    "HYBRID_ALPHA",
    "MMR_LAMBDA",
    "SEARCH_LIMIT",
    "__version__",
]

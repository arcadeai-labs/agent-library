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
    libr --help
"""

__version__ = "0.11.0"

__all__ = [
    "__version__",
]

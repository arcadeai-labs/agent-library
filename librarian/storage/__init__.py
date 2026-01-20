"""
Storage module for librarian.

Provides SQLite-based storage with vector search (sqlite-vec) and
full-text search (FTS5) capabilities.

Usage:
    from librarian.storage import Database, get_database
    from librarian.storage import VectorStore, FTSStore
"""

from librarian.storage.database import Database, get_database
from librarian.storage.fts_store import FTSStore
from librarian.storage.vector_store import VectorStore
from librarian.types import Chunk, Document

__all__ = [
    "Chunk",
    "Database",
    "Document",
    "FTSStore",
    "VectorStore",
    "get_database",
]

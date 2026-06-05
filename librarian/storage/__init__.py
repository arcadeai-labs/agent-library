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
from librarian.storage.migrate import migrate
from librarian.storage.protocols import Storage, SyncState
from librarian.storage.sqlite_storage import SQLiteStorage, get_storage
from librarian.storage.vector_store import VectorStore
from librarian.types import Chunk, Document

__all__ = [
    "Chunk",
    "Database",
    "Document",
    "FTSStore",
    "SQLiteStorage",
    "Storage",
    "SyncState",
    "VectorStore",
    "get_database",
    "get_storage",
    "migrate",
]

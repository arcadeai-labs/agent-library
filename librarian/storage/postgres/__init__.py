"""
Postgres + pgvector storage backend (the v0.14 ``Storage`` bundle on Postgres).

Selected via ``STORAGE_BACKEND=postgres``; SQLite remains the default. See
:class:`librarian.storage.postgres.storage.PostgresStorage`.
"""

from librarian.storage.postgres.database import PostgresDatabase
from librarian.storage.postgres.fts_store import PgFTSStore
from librarian.storage.postgres.storage import PostgresStorage, get_postgres_storage
from librarian.storage.postgres.vector_store import PgVectorStore

__all__ = [
    "PgFTSStore",
    "PgVectorStore",
    "PostgresDatabase",
    "PostgresStorage",
    "get_postgres_storage",
]

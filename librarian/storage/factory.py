"""
Backend-neutral storage resolution.

``STORAGE_BACKEND`` (config / env) selects the concrete substrate behind the
``Storage`` protocol bundle: ``sqlite`` (default, zero external deps) or
``postgres`` (pgvector; requires the ``postgres`` extra). Callers obtain storage
through these helpers instead of importing a concrete backend, so the
orchestrator, retrieval, CLI and MCP layers stay substrate-agnostic.

* :func:`get_storage` -- a migrated bundle for write paths (ingest).
* :func:`get_read_storage` -- a bundle for read paths (search/metadata) without
  forcing a migration on every call.
* :func:`get_metadata_store` -- just the metadata capability store.

The config value is read dynamically (not import-bound) so tests and the CLI can
flip the backend per process.
"""

from typing import TYPE_CHECKING

import librarian.config as config

if TYPE_CHECKING:
    from librarian.storage.protocols import MetadataStore, Storage

__all__ = ["get_metadata_store", "get_read_storage", "get_storage"]


def _is_postgres() -> bool:
    return config.STORAGE_BACKEND == "postgres"


def get_storage() -> "Storage":
    """Return a migrated storage bundle for the active backend (write paths)."""
    if _is_postgres():
        from librarian.storage.postgres import get_postgres_storage

        return get_postgres_storage()

    from librarian.storage.database import get_database
    from librarian.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(database=get_database())
    storage.migrate()
    return storage


def get_read_storage() -> "Storage":
    """Return a storage bundle for read paths.

    The Postgres backend reuses its migrated process-wide instance; the SQLite
    backend wraps the current global database without re-running the migration
    (ingest is responsible for migrating).
    """
    if _is_postgres():
        from librarian.storage.postgres import get_postgres_storage

        return get_postgres_storage()

    from librarian.storage.database import get_database
    from librarian.storage.sqlite_storage import SQLiteStorage

    return SQLiteStorage(database=get_database())


def get_metadata_store() -> "MetadataStore":
    """Return the active backend's metadata (document read) store."""
    return get_read_storage().metadata

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

from collections.abc import Callable
from typing import TYPE_CHECKING

import librarian.config as config

if TYPE_CHECKING:
    from librarian.storage.protocols import MetadataStore, Storage

__all__ = ["get_metadata_store", "get_read_storage", "get_storage"]


# ---------------------------------------------------------------------------
# Per-backend builders
# ---------------------------------------------------------------------------
# Each backend registers a (write, read) builder pair. A new backend is added by
# writing two builders and one registry entry -- get_storage / get_read_storage
# never change (they're closed against new backends, open for extension).


def _sqlite_write() -> "Storage":
    from librarian.storage.database import get_database
    from librarian.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(database=get_database())
    storage.migrate()
    return storage


def _sqlite_read() -> "Storage":
    # Read path wraps the current global database without re-running the
    # migration (ingest owns migrating the SQLite file).
    from librarian.storage.database import get_database
    from librarian.storage.sqlite_storage import SQLiteStorage

    return SQLiteStorage(database=get_database())


def _postgres() -> "Storage":
    # Postgres reuses one migrated, process-wide instance for both read and write
    # paths (re-running its idempotent migration would be a wasteful no-op), so
    # the same builder backs both registries.
    from librarian.storage.postgres import get_postgres_storage

    return get_postgres_storage()


# backend name -> builder. Read and write differ only for SQLite (see above).
_WRITE_BACKENDS: dict[str, Callable[[], "Storage"]] = {
    "sqlite": _sqlite_write,
    "postgres": _postgres,
}
_READ_BACKENDS: dict[str, Callable[[], "Storage"]] = {
    "sqlite": _sqlite_read,
    "postgres": _postgres,
}


def _resolve(registry: dict[str, Callable[[], "Storage"]]) -> "Storage":
    backend = config.STORAGE_BACKEND
    try:
        builder = registry[backend]
    except KeyError:  # pragma: no cover - config validation rejects this earlier
        raise ValueError(
            f"Unknown STORAGE_BACKEND {backend!r}; expected one of: {', '.join(sorted(registry))}."
        ) from None
    return builder()


def get_storage() -> "Storage":
    """Return a migrated storage bundle for the active backend (write paths)."""
    return _resolve(_WRITE_BACKENDS)


def get_read_storage() -> "Storage":
    """Return a storage bundle for read paths (no forced migration on SQLite)."""
    return _resolve(_READ_BACKENDS)


def get_metadata_store() -> "MetadataStore":
    """Return the active backend's metadata (document read) store."""
    return get_read_storage().metadata

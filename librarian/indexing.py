"""
Indexing service (deprecated shim).

As of v0.14 the ingest pipeline lives in :class:`librarian.orchestrator.Orchestrator`,
driven by connectors (the built-in :class:`~librarian.connectors.LocalFileConnector`
for files). ``IndexingService`` is retained as a thin, backward-compatible shim:
``index_file`` constructs a one-shot ``LocalFileConnector`` + ``Orchestrator`` per
call and emits a ``DeprecationWarning`` pointing high-volume callers at the
``Orchestrator`` directly. It is slated for removal in v0.15.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

from librarian.processing.parsers.base import FileReadError, FileReadTimeoutError

logger = logging.getLogger(__name__)

_DEPRECATION_MESSAGE = (
    "IndexingService is deprecated and will be removed in v0.15. "
    "Use librarian.orchestrator.Orchestrator with a connector "
    "(e.g. LocalFileConnector) instead."
)


def _build_orchestrator() -> Any:
    """Construct an Orchestrator bound to the active storage backend (v0.14 schema)."""
    from librarian.orchestrator import Orchestrator
    from librarian.storage.factory import get_storage

    return Orchestrator(storage=get_storage())


class IndexingService:
    """Deprecated thin shim over :class:`~librarian.orchestrator.Orchestrator`."""

    def __init__(self) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

    def index_file(self, file_path: Path) -> dict[str, Any]:
        """Index a single file. Deprecated: use ``Orchestrator`` directly.

        Raises:
            FileReadTimeoutError: If file stat times out (cloud storage not synced).
            FileReadError: For I/O errors accessing the file.
            FileNotFoundError: If the file doesn't exist.
        """
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        result: dict[str, Any] = _build_orchestrator().index_file(file_path)
        return result

    def should_reindex(self, file_path: Path) -> bool:
        """Check whether a file needs reindexing based on modification time.

        Raises:
            FileReadTimeoutError: If stat() times out.
            FileReadError: For I/O errors.
        """
        # Route through the storage factory so this honors STORAGE_BACKEND (it's a
        # normal ingest-path read, not SQLite-only maintenance).
        from librarian.storage.factory import get_metadata_store

        metadata = get_metadata_store()
        try:
            current_mtime = file_path.stat().st_mtime
        except TimeoutError as e:
            raise FileReadTimeoutError(
                f"Timed out accessing {file_path} (file may not be synced from cloud storage)"
            ) from e
        except OSError as e:
            raise FileReadError(f"Cannot access {file_path}: {e}") from e

        existing = metadata.get_document_by_path(str(file_path))
        if not existing:
            return True
        if existing.file_mtime is None:
            return True
        return current_mtime > existing.file_mtime


_indexing_service: IndexingService | None = None


def get_indexing_service() -> IndexingService:
    """Get the global indexing service instance (deprecated; see ``Orchestrator``)."""
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = IndexingService()
    return _indexing_service

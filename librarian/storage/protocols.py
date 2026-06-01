"""
Storage protocol bundle for the v0.14 single forward path.

A storage backend is described by four capability protocols that together form
the ``Storage`` bundle:

* :class:`MetadataStore` -- document metadata read access.
* :class:`VectorStore` -- vector similarity search.
* :class:`FTSStore` -- full-text (keyword) search.
* :class:`StateStore` -- connector sync cursors (the ``sync_state`` table).

The concrete :class:`~librarian.storage.sqlite_storage.SQLiteStorage` implements
the bundle. A future ``PostgresStorage`` (DEV-472) implements the same protocols
so callers are substrate-agnostic. The existing ``Database``, ``VectorStore`` and
``FTSStore`` classes satisfy the read-side protocols where applicable.

These are structural ``typing.Protocol`` definitions, so any object with the
right methods satisfies them -- no explicit subclassing required.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from librarian.storage.fts_store import FTSSearchResult
    from librarian.storage.vector_store import VectorSearchResult
    from librarian.types import Document

__all__ = [
    "FTSStore",
    "MetadataStore",
    "StateStore",
    "Storage",
    "SyncState",
    "VectorStore",
]


@dataclass
class SyncState:
    """One row of the ``sync_state`` table: a connector source's cursor + stats.

    ``source_key`` identifies a single sync stream (typically the connector
    name, optionally namespaced per source). ``cursor`` is the opaque
    JSON-serializable checkpoint the connector understands.
    """

    source_key: str
    cursor: dict[str, Any] = field(default_factory=dict)
    status: str = "idle"
    last_success_at: datetime | None = None
    last_attempt_at: datetime | None = None
    last_error: str | None = None
    documents_seen: int = 0
    chunks_written: int = 0
    config_version: int = 0


@runtime_checkable
class MetadataStore(Protocol):
    """Read access to document metadata."""

    def get_document_by_path(self, path: str) -> "Document | None": ...

    def get_document_by_id(self, doc_id: int) -> "Document | None": ...

    def list_documents(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> "list[Document]": ...

    def get_stats(self) -> dict[str, Any]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector similarity search."""

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> "list[VectorSearchResult]": ...


@runtime_checkable
class FTSStore(Protocol):
    """Full-text (keyword) search."""

    def search(
        self,
        query: str,
        limit: int = 10,
        snippet_length: int = 64,
    ) -> "list[FTSSearchResult]": ...


@runtime_checkable
class StateStore(Protocol):
    """Connector sync-cursor persistence (the ``sync_state`` table)."""

    def get_sync_state(self, source_key: str) -> SyncState | None: ...

    def put_sync_state(self, state: SyncState, conn: Any = None) -> None: ...


@runtime_checkable
class Storage(Protocol):
    """The full storage bundle a connector + orchestrator runs against.

    Bundles the four capability protocols and adds:

    * :meth:`migrate` -- create/upgrade librarian-owned tables.
    * :meth:`transaction` -- a context manager yielding a connection/handle on
      which content writes and cursor advances happen atomically.
    """

    metadata: MetadataStore
    vectors: VectorStore
    fts: FTSStore
    state: StateStore

    def migrate(self) -> None: ...

    def transaction(self) -> AbstractContextManager[Any]: ...

    def get_sync_state(self, source_key: str) -> SyncState | None: ...

    def put_sync_state(self, state: SyncState, conn: Any = None) -> None: ...

    def write_upsert(self, conn: Any, prepared: Any) -> None: ...

    def soft_delete_document(self, conn: Any, document_id: str, reason: str | None) -> int: ...


# Re-export Iterator for implementers that annotate transaction() precisely.
_ = Iterator

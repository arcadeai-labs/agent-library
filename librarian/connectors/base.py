"""
Connector contract for the v0.14 single forward path.

A ``Connector`` is a stateless, database-free source adapter. It knows how to
turn an external source (the local filesystem, Slack, GitHub, Linear, ...) into
an async stream of :class:`ChangeEvent` objects. It never touches storage,
embeddings, or chunking -- that is the :class:`~librarian.orchestrator.Orchestrator`'s
job. This keeps connector unit tests trivial: feed an input fixture, assert the
emitted events, no database required.

The contract is intentionally tiny:

* :meth:`Connector.initial_state` -- the cursor to start from when nothing has
  been synced yet.
* :meth:`Connector.fetch_changes` -- given the last persisted cursor, yield the
  changes since then as an ``AsyncIterator[ChangeEvent]``.

Each event may carry an opaque ``checkpoint`` (JSON-serializable) describing the
cursor position *after* that event. The orchestrator persists the checkpoint in
the same transaction as the event's content, which is what makes a crash
mid-stream resumable without duplicating or skipping work.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from librarian.types import AssetType

__all__ = [
    "ChangeEvent",
    "ChunkInput",
    "Connector",
    "DocumentSoftDelete",
    "DocumentUpsert",
]


@dataclass
class ChunkInput:
    """A pre-formed chunk supplied by a connector.

    Conversational sources (Slack, Linear comments) chunk at the source: one
    message is one chunk. They emit ``ChunkInput`` objects directly and the
    orchestrator embeds + stores them as-is. File sources instead supply
    ``raw_content`` on the :class:`DocumentUpsert` and let the orchestrator route
    through the parser registry to produce chunks.
    """

    content: str
    chunk_index: int
    # Uniquely identifies this chunk within the connector's source. Combined with
    # the connector name + source type to derive the deterministic chunk id.
    source_native_id: str
    heading_path: str | None = None
    chunk_source_uri: str | None = None
    asset_type: AssetType = AssetType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentUpsert:
    """Create-or-replace a document and its chunks.

    Exactly one of ``chunks`` (pre-formed) or ``raw_content`` (routed through the
    parser registry) should be provided.
    """

    source_type: str
    # Uniquely identifies the document within the connector's source. Combined
    # with the connector name + source type to derive the deterministic doc id.
    source_native_id: str
    asset_type: AssetType = AssetType.TEXT
    title: str | None = None
    document_source_uri: str | None = None
    # File sources: raw bytes/str + mimetype, routed through the parser registry.
    raw_content: str | bytes | None = None
    mimetype: str | None = None
    # Conversational sources: pre-formed chunks.
    chunks: list[ChunkInput] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_created_at: datetime | None = None
    # Opaque cursor describing the position *after* this event. Persisted
    # atomically with the event's content by the orchestrator.
    checkpoint: dict[str, Any] | None = None


@dataclass
class DocumentSoftDelete:
    """Tombstone a document's chunks without removing the rows.

    Used for meaningful structural deletes (an issue archived, a page removed).
    The chunks stay in the corpus with ``deleted_at`` / ``deletion_reason`` set so
    institutional memory survives source-side deletes.
    """

    source_type: str
    source_native_id: str
    deletion_reason: str | None = None
    checkpoint: dict[str, Any] | None = None


# A change event is either an upsert or a soft delete.
ChangeEvent = DocumentUpsert | DocumentSoftDelete


class Connector(ABC):
    """Abstract base class for all source connectors.

    Subclasses must set :attr:`name` (used as the namespace for deterministic
    id hashing) and implement :meth:`initial_state` and :meth:`fetch_changes`.

    Connectors are stateless with respect to storage: they receive the last
    persisted cursor as an argument and return changes. They must not open
    database connections, compute embeddings, or chunk file content.
    """

    #: Stable connector name, used in deterministic id hashing. Override in
    #: subclasses (e.g. ``"local_file"``, ``"slack"``).
    name: str = "connector"

    @abstractmethod
    def initial_state(self) -> dict[str, Any]:
        """Return the starting cursor for a source that has never been synced.

        The value is opaque JSON understood only by this connector. The
        orchestrator persists it and hands it back to :meth:`fetch_changes`.
        """
        ...

    @abstractmethod
    def fetch_changes(self, state: dict[str, Any]) -> AsyncIterator[ChangeEvent]:
        """Yield changes since the given cursor.

        Args:
            state: The last persisted cursor (or :meth:`initial_state` on first
                run).

        Returns:
            An async iterator of :class:`ChangeEvent`. Implementations are
            typically ``async def`` generators.
        """
        ...

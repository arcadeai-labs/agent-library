"""
Orchestrator integration tests.

A fixture connector emits a deterministic stream of ``ChangeEvent``s; the
orchestrator processes them against a real (temp-file) SQLite substrate with a
fake embedder. These tests assert the tracer-bullet acceptance criteria:

* chunks land in the new schema with deterministic ids;
* re-running is idempotent (no duplicate chunks);
* content + cursor advance commit in one transaction (a simulated mid-stream
  crash leaves no orphan ``chunks`` rows and no cursor past uncommitted work);
* a soft delete tombstones chunks without removing rows.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from librarian.connectors import ChunkInput, Connector, DocumentSoftDelete, DocumentUpsert
from librarian.orchestrator import Orchestrator
from librarian.storage.database import Database
from librarian.storage.sqlite_storage import SQLiteStorage
from librarian.types import AssetType

from .conftest import FakeEmbedder


class FakeConnector(Connector):
    """Yields a fixed list of events, threading a simple integer cursor."""

    name = "fake"

    def __init__(self, events: list) -> None:
        self._events = events

    def initial_state(self) -> dict:
        return {"i": 0}

    async def fetch_changes(self, state: dict) -> AsyncIterator:
        start = int(state.get("i", 0))
        for idx in range(start, len(self._events)):
            event = self._events[idx]
            event.checkpoint = {"i": idx + 1}
            yield event


def _upsert(native_id: str, text: str) -> DocumentUpsert:
    return DocumentUpsert(
        source_type="msg",
        source_native_id=native_id,
        asset_type=AssetType.TEXT,
        title=native_id,
        chunks=[
            ChunkInput(
                content=text,
                chunk_index=0,
                source_native_id=f"{native_id}#0",
                asset_type=AssetType.TEXT,
            )
        ],
    )


def _multi_upsert(native_id: str, texts: list[str]) -> DocumentUpsert:
    return DocumentUpsert(
        source_type="msg",
        source_native_id=native_id,
        asset_type=AssetType.TEXT,
        title=native_id,
        chunks=[
            ChunkInput(
                content=text,
                chunk_index=i,
                source_native_id=f"{native_id}#{i}",
                asset_type=AssetType.TEXT,
            )
            for i, text in enumerate(texts)
        ],
    )


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records every chunk it is asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        self.embedded.extend(documents)
        return super().embed_documents(documents)


@pytest.fixture
def storage(tmp_path: Path) -> SQLiteStorage:
    db = Database(str(tmp_path / "orch.db"))
    st = SQLiteStorage(database=db)
    st.migrate()
    return st


def _count(storage: SQLiteStorage, table: str) -> int:
    conn = storage.database._get_connection()
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608


def _chunk_ids(storage: SQLiteStorage) -> list[str]:
    conn = storage.database._get_connection()
    return [r[0] for r in conn.execute("SELECT chunk_id FROM chunks ORDER BY id").fetchall()]


async def test_sync_writes_chunks_and_advances_cursor(storage: SQLiteStorage) -> None:
    connector = FakeConnector([_upsert("m1", "hello"), _upsert("m2", "world")])
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())

    result = await orch.sync(connector)

    assert result.documents_upserted == 2
    assert result.chunks_written == 2
    assert _count(storage, "chunks") == 2
    assert _count(storage, "documents") == 2
    assert _count(storage, "chunk_embeddings") == 2

    state = storage.get_sync_state("fake")
    assert state is not None
    assert state.cursor == {"i": 2}

    # chunk ids are the deterministic hash, not integers.
    cids = _chunk_ids(storage)
    assert all(isinstance(c, str) and len(c) == 64 for c in cids)


async def test_reingest_is_idempotent(storage: SQLiteStorage) -> None:
    events = [_upsert("m1", "hello"), _upsert("m2", "world")]
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())

    await orch.sync(FakeConnector(events))
    ids_first = set(_chunk_ids(storage))

    # Re-run from scratch (fresh cursor) over the same content.
    storage.put_sync_state(_reset_state())
    await orch.sync(FakeConnector(events))

    assert _count(storage, "chunks") == 2  # no duplicates
    assert set(_chunk_ids(storage)) == ids_first  # same deterministic ids


async def test_crash_midstream_leaves_no_orphans_and_resumes(storage: SQLiteStorage) -> None:
    events = [_upsert("m1", "a"), _upsert("m2", "b"), _upsert("m3", "c")]
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())

    # Make the 3rd content write blow up, simulating a crash mid-stream.
    original = storage.write_upsert
    calls = {"n": 0}

    def exploding_write(conn, prepared):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        return original(conn, prepared)

    storage.write_upsert = exploding_write  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated crash"):
        await orch.sync(FakeConnector(events))

    # The 3rd event's content AND cursor advance both rolled back: 2 chunks, cursor at 2.
    assert _count(storage, "chunks") == 2
    state = storage.get_sync_state("fake")
    assert state is not None and state.cursor == {"i": 2}

    # Recover and resume from the persisted cursor: no skipped or duplicated work.
    storage.write_upsert = original  # type: ignore[method-assign]
    result = await orch.sync(FakeConnector(events))

    assert result.documents_upserted == 1  # only the remaining event
    assert _count(storage, "chunks") == 3
    assert storage.get_sync_state("fake").cursor == {"i": 3}  # type: ignore[union-attr]


async def test_crash_during_cursor_advance_rolls_back_chunks(storage: SQLiteStorage) -> None:
    """The inverse of the write-side crash: blow up *during* the cursor advance.

    This is the test that actually proves the single-transaction binding. By the
    time ``put_sync_state`` runs, ``write_upsert`` has already inserted the
    document, chunk, and embedding rows on the same connection. If they shared a
    transaction with the cursor advance, the failure must roll *all* of it back:
    no orphan ``chunks`` rows left behind, and no ``sync_state`` row pointing
    past content that never committed.
    """
    events = [_upsert("m1", "a"), _upsert("m2", "b")]
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())

    # Let the first event's cursor advance succeed; explode on the second.
    original = storage.put_sync_state
    calls = {"n": 0}

    def exploding_put_sync_state(state, conn=None):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash during cursor advance")
        return original(state, conn=conn)

    storage.put_sync_state = exploding_put_sync_state  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated crash during cursor advance"):
        await orch.sync(FakeConnector(events))

    storage.put_sync_state = original  # type: ignore[method-assign]

    # The second event's chunks were inserted before the crash, but the failed
    # cursor advance rolled them back along with the document and embedding rows.
    assert _count(storage, "chunks") == 1
    assert _count(storage, "documents") == 1
    assert _count(storage, "chunk_embeddings") == 1  # no orphan embeddings either

    # The sync_state row still exists (from event 1) but was not advanced past
    # the content that actually committed -- no orphan cursor.
    state = storage.get_sync_state("fake")
    assert state is not None
    assert state.cursor == {"i": 1}

    # Recover and resume: the second event is re-emitted and lands cleanly.
    result = await orch.sync(FakeConnector(events))
    assert result.documents_upserted == 1
    assert _count(storage, "chunks") == 2
    assert storage.get_sync_state("fake").cursor == {"i": 2}  # type: ignore[union-attr]


async def test_crash_on_first_cursor_advance_leaves_no_sync_state_row(
    storage: SQLiteStorage,
) -> None:
    """A crash on the very first event must leave the DB completely pristine.

    No partially-written chunks, and crucially no ``sync_state`` row at all --
    an orphan cursor with zero committed content would make the source look
    "started" while nothing was persisted.
    """
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())

    def boom(state, conn=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated crash on first advance")

    storage.put_sync_state = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated crash on first advance"):
        await orch.sync(FakeConnector([_upsert("m1", "a")]))

    assert _count(storage, "chunks") == 0
    assert _count(storage, "documents") == 0
    assert _count(storage, "chunk_embeddings") == 0
    assert storage.get_sync_state("fake") is None  # no orphan sync_state row


async def test_soft_delete_tombstones_without_removing_rows(storage: SQLiteStorage) -> None:
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())
    await orch.sync(FakeConnector([_upsert("m1", "hello")]))
    assert _count(storage, "chunks") == 1

    delete_event = DocumentSoftDelete(
        source_type="msg", source_native_id="m1", deletion_reason="gone"
    )
    await orch.sync(FakeConnector([delete_event]), source_key="fake-del")

    # Row is still present, but tombstoned.
    assert _count(storage, "chunks") == 1
    conn = storage.database._get_connection()
    deleted_at, reason = conn.execute(
        "SELECT deleted_at, deletion_reason FROM chunks LIMIT 1"
    ).fetchone()
    assert deleted_at is not None
    assert reason == "gone"


async def test_soft_delete_hides_chunk_from_search(storage: SQLiteStorage) -> None:
    """Tombstoned chunks must drop out of BOTH keyword and semantic search.

    Verifying only the column state (as ``test_soft_delete_tombstones_...`` does)
    would miss that the read paths must filter ``deleted_at IS NULL`` -- without
    that filter a soft-deleted chunk stays fully searchable.
    """
    embedder = FakeEmbedder()
    orch = Orchestrator(storage=storage, embedder=embedder)
    await orch.sync(FakeConnector([_upsert("m1", "hello world")]))

    # Query with the exact vector the chunk was stored under so the semantic hit
    # is a guaranteed self-match (FakeEmbedder vectors are otherwise unrelated to
    # the query text), isolating what we are testing: the tombstone filter.
    query_embedding = embedder.embed_documents(["hello world"])[0]

    # Visible before the delete via both modalities.
    assert storage.fts.search("hello") != []
    assert storage.vectors.search(query_embedding) != []

    delete_event = DocumentSoftDelete(
        source_type="msg", source_native_id="m1", deletion_reason="gone"
    )
    await orch.sync(FakeConnector([delete_event]), source_key="fake-del")

    # Gone from both keyword (FTS) and semantic (vector) search.
    assert storage.fts.search("hello") == []
    assert storage.vectors.search(query_embedding) == []


async def test_reingest_only_reembeds_changed_chunks(storage: SQLiteStorage) -> None:
    """Editing one chunk of a multi-chunk document re-embeds only that chunk."""
    embedder = CountingEmbedder()
    orch = Orchestrator(storage=storage, embedder=embedder)

    await orch.sync(FakeConnector([_multi_upsert("d1", ["alpha", "beta"])]))
    assert embedder.embedded == ["alpha", "beta"]  # both embedded on first ingest

    embedder.embedded.clear()

    # Re-run from scratch over the same document with one chunk changed.
    storage.put_sync_state(_reset_state())
    await orch.sync(FakeConnector([_multi_upsert("d1", ["alpha", "gamma"])]))

    # Only the changed chunk is re-embedded; the unchanged one reuses its vector.
    assert embedder.embedded == ["gamma"]
    assert _count(storage, "chunks") == 2


def _reset_state():  # type: ignore[no-untyped-def]
    from librarian.storage.protocols import SyncState

    return SyncState(source_key="fake", cursor={"i": 0})

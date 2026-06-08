"""
Substrate-parity contract suite for the v0.14 ``Storage`` bundle.

The same assertions run against every backend via the parameterized ``backend``
fixture: today SQLite (always) and Postgres (when a test database is reachable
through ``TEST_POSTGRES_DSN``; otherwise skipped). This is the executable form
of the DEV-472 acceptance criteria -- the storage abstraction is only
"genuinely pluggable" if one suite passes unchanged on two substrates.

Each backend is wrapped in a small ``StorageHarness`` so the tests can make
backend-neutral row-count assertions with identical SQL. Postgres runs are
isolated in a throwaway schema per test so commits/rollbacks are exercised for
real (no transaction-wrapping that would mask the atomicity guarantees).
"""

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from librarian import ids
from librarian.storage.write_models import PreparedChunk, PreparedDocument
from librarian.types import AssetType, EmbeddingModality

from .conftest import FakeEmbedder

CONNECTOR = "contract"
SOURCE_TYPE = "doc"
EMBED_DIM = 384  # matches conftest's EMBEDDING_DIMENSION / FakeEmbedder


@dataclass
class StorageHarness:
    """A storage bundle plus backend-neutral inspection helpers."""

    storage: Any
    backend: str

    def count(self, table: str, where: str | None = None) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608 - fixed table literals
        if where:
            sql += f" WHERE {where}"
        with self.storage.database._connection() as conn:
            return conn.execute(sql).fetchone()["n"]

    def chunk_ids(self) -> list[str]:
        with self.storage.database._connection() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks ORDER BY id").fetchall()
        return [row["chunk_id"] for row in rows]


# =============================================================================
# Fixtures
# =============================================================================


def _make_sqlite(tmp_path: Path) -> StorageHarness:
    from librarian.storage.database import Database
    from librarian.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(database=Database(str(tmp_path / "contract.db")))
    storage.migrate()
    return StorageHarness(storage=storage, backend="sqlite")


def _make_postgres(dsn: str) -> Iterator[StorageHarness]:
    from librarian.storage.postgres import PostgresStorage

    schema = f"librarian_test_{uuid.uuid4().hex}"
    storage = PostgresStorage(dsn=dsn, schema=schema)
    storage.migrate()
    try:
        yield StorageHarness(storage=storage, backend="postgres")
    finally:
        from psycopg import sql

        conn = storage.database._get_connection()
        try:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
            conn.commit()
        finally:
            storage.database.close()


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageHarness]:
    if request.param == "sqlite":
        yield _make_sqlite(tmp_path)
        return

    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set; skipping Postgres parity run")
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip("psycopg not installed; skipping Postgres parity run")
    yield from _make_postgres(dsn)


# =============================================================================
# Helpers
# =============================================================================


def _prepare(native_id: str, text: str, embedder: FakeEmbedder) -> PreparedDocument:
    """Build a single-chunk PreparedDocument with deterministic ids."""
    chunk_native = f"{native_id}#0"
    embedding = embedder.embed_documents([text])[0]
    chunk = PreparedChunk(
        chunk_id=ids.chunk_id(CONNECTOR, SOURCE_TYPE, chunk_native),
        content=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        asset_type=AssetType.TEXT,
        modality=EmbeddingModality.TEXT,
        embedding=embedding,
        model_version=embedder.model_name,
    )
    return PreparedDocument(
        document_id=ids.document_id(CONNECTOR, SOURCE_TYPE, native_id),
        path=native_id,
        title=native_id,
        content=text,
        metadata={"source": "contract"},
        asset_type=AssetType.TEXT,
        document_size=len(text),
        chunks=[chunk],
    )


def _write(harness: StorageHarness, prepared: PreparedDocument) -> None:
    with harness.storage.transaction() as conn:
        harness.storage.write_upsert(conn, prepared)


# =============================================================================
# Contract tests (run against every backend)
# =============================================================================


def test_write_upsert_persists_document_and_chunks(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1
    assert backend.count("chunk_embeddings") == 1

    doc = backend.storage.metadata.get_document_by_path("m1")
    assert doc is not None
    assert doc.content == "hello world"
    assert doc.metadata.get("source") == "contract"


def test_reingest_is_idempotent(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)

    _write(backend, prepared)
    ids_first = set(backend.chunk_ids())
    _write(backend, _prepare("m1", "hello world", embedder))

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1  # replaced, not duplicated
    assert backend.count("chunk_embeddings") == 1
    assert set(backend.chunk_ids()) == ids_first  # same deterministic ids


def test_existing_text_chunks_returns_live_text_embeddings(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)
    _write(backend, prepared)

    existing = backend.storage.existing_text_chunks(prepared.document_id)

    assert set(existing) == {prepared.chunks[0].chunk_id}
    content, embedding = existing[prepared.chunks[0].chunk_id]
    assert content == "hello world"
    assert embedding == pytest.approx(prepared.chunks[0].embedding)


def test_write_and_cursor_advance_commit_together(backend: StorageHarness) -> None:
    from librarian.storage.protocols import SyncState

    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)

    with backend.storage.transaction() as conn:
        backend.storage.write_upsert(conn, prepared)
        backend.storage.put_sync_state(
            SyncState(source_key="contract", cursor={"i": 1}, status="ok"),
            conn=conn,
        )

    assert backend.count("chunks") == 1
    state = backend.storage.get_sync_state("contract")
    assert state is not None
    assert state.cursor == {"i": 1}


def test_crash_rolls_back_content_and_cursor(backend: StorageHarness) -> None:
    """The transactional acceptance test: a crash after the content write but
    before the commit must roll *both* the content and the cursor advance back."""
    from librarian.storage.protocols import SyncState

    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)

    with pytest.raises(RuntimeError, match="boom"), backend.storage.transaction() as conn:
        backend.storage.write_upsert(conn, prepared)
        backend.storage.put_sync_state(SyncState(source_key="contract", cursor={"i": 1}), conn=conn)
        raise RuntimeError("boom")  # simulated crash before commit

    # Nothing committed: no orphan rows, no orphan cursor.
    assert backend.count("documents") == 0
    assert backend.count("chunks") == 0
    assert backend.count("chunk_embeddings") == 0
    assert backend.storage.get_sync_state("contract") is None


def test_soft_delete_tombstones_without_removing_rows(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)
    _write(backend, prepared)

    with backend.storage.transaction() as conn:
        tombstoned = backend.storage.soft_delete_document(conn, prepared.document_id, "gone")

    assert tombstoned == 1
    assert backend.count("chunks") == 1  # row still present
    assert backend.count("chunks", "deleted_at IS NOT NULL") == 1
    assert backend.count("chunks", "deletion_reason = 'gone'") == 1


def test_sync_state_roundtrip(backend: StorageHarness) -> None:
    from librarian.storage.protocols import SyncState

    assert backend.storage.get_sync_state("nope") is None

    backend.storage.put_sync_state(
        SyncState(
            source_key="contract",
            cursor={"mtimes": {"/a": 1.5}},
            status="ok",
            documents_seen=3,
            chunks_written=7,
            config_version=2,
        )
    )
    state = backend.storage.get_sync_state("contract")
    assert state is not None
    assert state.cursor == {"mtimes": {"/a": 1.5}}
    assert state.documents_seen == 3
    assert state.chunks_written == 7
    assert state.config_version == 2

    # Upsert overwrites in place (no duplicate row).
    backend.storage.put_sync_state(SyncState(source_key="contract", cursor={"i": 9}, status="ok"))
    assert backend.count("sync_state") == 1
    assert backend.storage.get_sync_state("contract").cursor == {"i": 9}  # type: ignore[union-attr]


def test_file_mtime_roundtrip(backend: StorageHarness) -> None:
    assert backend.storage.state.get_file_mtime("contract", "/tmp/doc.md") is None

    backend.storage.state.set_file_mtime("contract", "/tmp/doc.md", 123.456)
    assert backend.storage.state.get_file_mtime("contract", "/tmp/doc.md") == pytest.approx(
        123.456
    )

    with backend.storage.transaction() as conn:
        backend.storage.state.set_file_mtime("contract", "/tmp/doc.md", 789.0, conn=conn)

    assert backend.storage.state.get_file_mtime("contract", "/tmp/doc.md") == pytest.approx(789.0)


def test_vector_search_finds_written_chunk(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "the quick brown fox", embedder))
    _write(backend, _prepare("m2", "lorem ipsum dolor", embedder))

    # Query with m1's own (deterministic) document embedding so the nearest
    # neighbor is unambiguous across substrates -- FakeEmbedder vectors are
    # otherwise uncorrelated random noise.
    query = embedder.embed_documents(["the quick brown fox"])[0]
    results = backend.storage.vectors.search(query, limit=5, min_similarity=-1.0)

    assert results
    assert results[0].document_path == "m1"
    # similarity = 1 - distance is in a sane cosine range, best match first.
    assert all(-1.01 <= (1.0 - r.distance) <= 1.01 for r in results)
    assert (1.0 - results[0].distance) == pytest.approx(1.0, abs=1e-3)


def test_fts_search_finds_written_chunk(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "the quick brown fox", embedder))
    _write(backend, _prepare("m2", "lorem ipsum dolor", embedder))

    results = backend.storage.fts.search("quick fox", limit=5)

    assert results
    assert any(r.document_path == "m1" for r in results)


def test_migrate_is_idempotent(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))

    # Re-running migrate on a populated database is a no-op (no data loss).
    backend.storage.migrate()
    backend.storage.migrate()

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1
    assert backend.count("chunk_embeddings") == 1


def test_stats_reports_counts(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))

    stats = backend.storage.metadata.get_stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 1
    assert stats["embedding_count"] == 1

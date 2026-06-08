"""
PostgresStorage -- the pgvector-backed implementation of the v0.14 Storage bundle.

A drop-in peer of :class:`librarian.storage.sqlite_storage.SQLiteStorage`: it
satisfies the same ``Storage`` protocol (``metadata`` / ``vectors`` / ``fts`` /
``state`` capability stores plus ``migrate`` / ``transaction`` / write paths) so
the orchestrator and retrieval layer run against it unchanged.

The crucial guarantee carries over verbatim: :meth:`write_upsert` and
:meth:`put_sync_state` issued inside one :meth:`transaction` block commit or
roll back together, so a mid-stream crash never leaves orphan ``chunks`` rows
or a cursor pointing past uncommitted content.
"""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from librarian.config import POSTGRES_DSN, POSTGRES_SCHEMA
from librarian.storage.postgres.database import (
    PostgresDatabase,
    _json_default,
    parse_vector,
    vector_literal,
)
from librarian.storage.postgres.fts_store import PgFTSStore
from librarian.storage.postgres.migrate import migrate as migrate_schema
from librarian.storage.postgres.vector_store import PgVectorStore
from librarian.storage.protocols import SyncState
from librarian.storage.write_models import PreparedDocument
from librarian.types import EmbeddingModality

logger = logging.getLogger(__name__)

__all__ = ["PostgresStorage", "get_postgres_storage"]


def _modality_table(modality: EmbeddingModality) -> str:
    if modality == EmbeddingModality.CODE:
        return "vec_chunks_code"
    if modality == EmbeddingModality.VISION:
        return "vec_chunks_vision"
    return "chunk_embeddings"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class PostgresStorage:
    """Postgres-backed implementation of the v0.14 ``Storage`` bundle."""

    def __init__(
        self,
        database: PostgresDatabase | None = None,
        dsn: str | None = None,
        schema: str | None = None,
    ) -> None:
        self._db = database or PostgresDatabase(
            dsn=dsn or POSTGRES_DSN, schema=schema or POSTGRES_SCHEMA
        )
        self.metadata = self._db
        self.vectors = PgVectorStore(self._db)
        self.fts = PgFTSStore(self._db)
        # The StateStore protocol is satisfied by this object itself.
        self.state = self

    @property
    def database(self) -> PostgresDatabase:
        return self._db

    def migrate(self) -> None:
        """Create/upgrade the v0.14 Postgres schema (idempotent)."""
        conn = self._db._get_connection()
        migrate_schema(conn, schema=self._db.schema)

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Yield the thread-local connection as one atomic transaction.

        Commits on clean exit, rolls back on exception. Content writes and
        cursor advances issued in the same ``with`` block commit or roll back
        together.
        """
        conn = self._db._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    # =========================================================================
    # StateStore
    # =========================================================================

    def get_sync_state(self, source_key: str) -> SyncState | None:
        with self._db._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sync_state WHERE source_key = %s", (source_key,)
            ).fetchone()
        if row is None:
            return None
        cursor = row["cursor"]
        if isinstance(cursor, str):
            cursor = json.loads(cursor) if cursor else {}
        elif cursor is None:
            cursor = {}
        return SyncState(
            source_key=row["source_key"],
            cursor=cursor,
            status=row["status"],
            last_success_at=row["last_success_at"],
            last_attempt_at=row["last_attempt_at"],
            last_error=row["last_error"],
            documents_seen=row["documents_seen"] or 0,
            chunks_written=row["chunks_written"] or 0,
            config_version=row["config_version"] or 0,
        )

    def put_sync_state(self, state: SyncState, conn: Any = None) -> None:
        """Upsert a sync-state row, joining ``conn``'s transaction when given."""
        own = conn is None
        conn = conn or self._db._get_connection()
        conn.execute(
            """
            INSERT INTO sync_state (
                source_key, cursor, status, last_success_at, last_attempt_at,
                last_error, documents_seen, chunks_written, config_version
            ) VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(source_key) DO UPDATE SET
                cursor = excluded.cursor,
                status = excluded.status,
                last_success_at = excluded.last_success_at,
                last_attempt_at = excluded.last_attempt_at,
                last_error = excluded.last_error,
                documents_seen = excluded.documents_seen,
                chunks_written = excluded.chunks_written,
                config_version = excluded.config_version
            """,
            (
                state.source_key,
                json.dumps(state.cursor),
                state.status,
                _iso(state.last_success_at)
                if isinstance(state.last_success_at, datetime)
                else state.last_success_at,
                _iso(state.last_attempt_at)
                if isinstance(state.last_attempt_at, datetime)
                else state.last_attempt_at,
                state.last_error,
                state.documents_seen,
                state.chunks_written,
                state.config_version,
            ),
        )
        if own:
            conn.commit()

    def get_file_mtime(self, source_key: str, path: str) -> float | None:
        """Return the last-recorded mtime for a single file, or ``None``."""
        with self._db._connection() as conn:
            row = conn.execute(
                "SELECT mtime FROM source_file_state WHERE source_key = %s AND path = %s",
                (source_key, path),
            ).fetchone()
        return float(row["mtime"]) if row is not None else None

    def set_file_mtime(
        self,
        source_key: str,
        path: str,
        mtime: float,
        conn: Any = None,
    ) -> None:
        """Upsert one file's mtime, joining ``conn``'s transaction when given."""
        own = conn is None
        conn = conn or self._db._get_connection()
        conn.execute(
            """
            INSERT INTO source_file_state (source_key, path, mtime)
            VALUES (%s, %s, %s)
            ON CONFLICT(source_key, path) DO UPDATE SET
                mtime = excluded.mtime,
                updated_at = now()
            """,
            (source_key, path, mtime),
        )
        if own:
            conn.commit()

    # =========================================================================
    # Write paths
    # =========================================================================

    def existing_text_chunks(self, document_id: str) -> dict[str, tuple[str, list[float] | None]]:
        """Return the document's live chunks as ``chunk_id -> (content, text_embedding)``.

        Mirrors SQLiteStorage so the orchestrator can reuse unchanged text
        embeddings before replacing a document's chunks.
        """
        conn = self._db._get_connection()
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.content, ce.embedding AS embedding
            FROM chunks c
            LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
            WHERE c.document_id = (SELECT id FROM documents WHERE document_id = %s)
              AND c.deleted_at IS NULL
              AND c.chunk_id IS NOT NULL
            """,
            (document_id,),
        ).fetchall()
        return {row["chunk_id"]: (row["content"], parse_vector(row["embedding"])) for row in rows}

    def write_upsert(self, conn: Any, prepared: PreparedDocument) -> None:
        """Create-or-replace a document and all its chunks within ``conn``'s txn."""
        doc_pk = self._upsert_document_row(conn, prepared)
        self._replace_chunks(conn, doc_pk, prepared)

    def _upsert_document_row(self, conn: Any, prepared: PreparedDocument) -> int:
        metadata_json = (
            json.dumps(prepared.metadata, default=_json_default) if prepared.metadata else None
        )
        row = conn.execute(
            "SELECT id FROM documents WHERE document_id = %s OR path = %s",
            (prepared.document_id, prepared.path),
        ).fetchone()
        if row is not None:
            doc_pk = row["id"]
            conn.execute(
                """
                UPDATE documents
                SET document_id = %s, path = %s, title = %s, content = %s,
                    metadata = %s::jsonb, file_mtime = %s, asset_type = %s,
                    document_source_uri = %s, source_created_at = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    prepared.document_id,
                    prepared.path,
                    prepared.title,
                    prepared.content,
                    metadata_json,
                    prepared.file_mtime,
                    prepared.asset_type.value,
                    prepared.document_source_uri,
                    _iso(prepared.source_created_at),
                    doc_pk,
                ),
            )
            # Chunk embeddings cascade from the chunks FK.
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_pk,))
            return int(doc_pk)

        row = conn.execute(
            """
            INSERT INTO documents (
                document_id, path, title, content, metadata, file_mtime,
                asset_type, document_source_uri, source_created_at
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                prepared.document_id,
                prepared.path,
                prepared.title,
                prepared.content,
                metadata_json,
                prepared.file_mtime,
                prepared.asset_type.value,
                prepared.document_source_uri,
                _iso(prepared.source_created_at),
            ),
        ).fetchone()
        return int(row["id"])

    def _replace_chunks(self, conn: Any, doc_pk: int, prepared: PreparedDocument) -> None:
        for chunk in prepared.chunks:
            row = conn.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_id, content, heading_path, chunk_index,
                    start_char, end_char, asset_type, modality, document_size,
                    source_created_at, document_source_uri, chunk_source_uri
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    doc_pk,
                    chunk.chunk_id,
                    chunk.content,
                    chunk.heading_path,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.asset_type.value,
                    chunk.modality.value,
                    prepared.document_size,
                    _iso(prepared.source_created_at),
                    prepared.document_source_uri,
                    chunk.chunk_source_uri,
                ),
            ).fetchone()
            chunk_pk = row["id"]
            if chunk.embedding:
                table = _modality_table(chunk.modality)
                literal = vector_literal(chunk.embedding)
                if table == "chunk_embeddings":
                    conn.execute(
                        "INSERT INTO chunk_embeddings (chunk_id, embedding, model_version) "
                        "VALUES (%s, %s::vector, %s)",
                        (chunk_pk, literal, chunk.model_version),
                    )
                else:
                    conn.execute(
                        f"INSERT INTO {table} (chunk_id, embedding) "  # noqa: S608
                        "VALUES (%s, %s::vector)",
                        (chunk_pk, literal),
                    )

    def soft_delete_document(self, conn: Any, document_id: str, reason: str | None) -> int:
        """Tombstone all chunks of a document. Returns the number tombstoned."""
        now = datetime.now().isoformat()
        cursor = conn.execute(
            """
            UPDATE chunks
            SET deleted_at = %s, deletion_reason = %s
            WHERE document_id = (SELECT id FROM documents WHERE document_id = %s)
              AND deleted_at IS NULL
            """,
            (now, reason, document_id),
        )
        return int(cursor.rowcount)


_storage_instance: PostgresStorage | None = None


def get_postgres_storage() -> PostgresStorage:
    """Get a process-wide :class:`PostgresStorage` (migrated to v0.14)."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = PostgresStorage()
        _storage_instance.migrate()
    return _storage_instance

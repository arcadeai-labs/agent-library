"""
PostgresDatabase -- connection management and read-side metadata access.

Mirrors the read surface of :class:`librarian.storage.database.Database` (the
SQLite metadata store) so the retrieval and MCP layers are substrate-agnostic.
Connections are thread-local with an ``RLock``, matching the SQLite manager's
concurrency model; each connection pins ``search_path`` to the configured
schema so multiple deployments (and isolated test runs) can share one database.

Embeddings are exchanged as pgvector text literals (``[1,2,3]``) and cast in
SQL (``%s::vector``). This keeps the backend working with only ``psycopg``
installed and sidesteps the extension-OID lookup ordering that
``pgvector.psycopg.register_vector`` requires on a freshly created database.
"""

import json
import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

from librarian.config import POSTGRES_DSN, POSTGRES_SCHEMA
from librarian.types import AssetType, Document

logger = logging.getLogger(__name__)

__all__ = ["PostgresDatabase", "vector_literal"]


def _require_psycopg() -> Any:
    """Import psycopg, raising a clear error if the ``postgres`` extra is absent."""
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PostgresStorage requires the 'postgres' extra. Install it with:\n"
            "    uv pip install -e '.[postgres]'\n"
            "or set STORAGE_BACKEND=sqlite to use the default backend."
        ) from e
    return psycopg


def vector_literal(embedding: list[float]) -> str:
    """Render an embedding as a pgvector text literal (``[0.1,0.2,...]``)."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def parse_vector(value: Any) -> list[float] | None:
    """Parse a pgvector value (text literal or list) back into floats."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip().strip("[]")
    if not text:
        return None
    return [float(x) for x in text.split(",")]


def _json_default(value: Any) -> str:
    """JSON fallback for date/datetime values from YAML frontmatter."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _as_metadata(value: Any) -> dict[str, Any]:
    """Normalize a JSONB column (psycopg returns dict) into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


class PostgresDatabase:
    """Postgres connection manager + read-side metadata store."""

    def __init__(self, dsn: str | None = None, schema: str | None = None) -> None:
        self.dsn = dsn or POSTGRES_DSN
        if not self.dsn:
            raise ValueError(
                "Postgres backend selected but no connection string configured. "
                "Set POSTGRES_DSN (or DATABASE_URL)."
            )
        self.schema = schema or POSTGRES_SCHEMA
        self._local = threading.local()
        self._lock = threading.RLock()

    # =========================================================================
    # Connection management
    # =========================================================================

    def _get_connection(self) -> Any:
        """Get or create a thread-local psycopg connection pinned to the schema."""
        conn = getattr(self._local, "connection", None)
        if conn is not None and not conn.closed:
            return conn

        psycopg = _require_psycopg()
        from psycopg import sql
        from psycopg.rows import dict_row

        conn = psycopg.connect(self.dsn, autocommit=False, row_factory=dict_row)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
            )
        conn.commit()
        self._local.connection = conn
        return conn

    @contextmanager
    def _connection(self) -> Generator[Any, None, None]:
        """Context manager that commits on clean exit and rolls back on error."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        """Close the current thread's connection, if any."""
        conn = getattr(self._local, "connection", None)
        if conn is not None and not conn.closed:
            conn.close()
        self._local.connection = None

    # =========================================================================
    # Document reads (MetadataStore protocol)
    # =========================================================================

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        asset_type = AssetType(row["asset_type"]) if row.get("asset_type") else AssetType.TEXT
        return Document(
            id=row["id"],
            path=row["path"],
            title=row["title"],
            content=row["content"],
            metadata=_as_metadata(row.get("metadata")),
            asset_type=asset_type,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            file_mtime=row.get("file_mtime"),
        )

    def get_document_by_path(self, path: str) -> Document | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE path = %s", (path,)).fetchone()
            return self._row_to_document(row) if row else None

    def get_document_by_id(self, doc_id: int) -> Document | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
            return self._row_to_document(row) if row else None

    def list_documents(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Document]:
        with self._connection() as conn:
            if start_date and end_date:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE updated_at >= %s AND updated_at < %s "
                    "ORDER BY updated_at DESC",
                    (start_date, end_date),
                ).fetchall()
            elif start_date:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE updated_at >= %s ORDER BY updated_at DESC",
                    (start_date,),
                ).fetchall()
            elif end_date:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE updated_at < %s ORDER BY updated_at DESC",
                    (end_date,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()
            return [self._row_to_document(row) for row in rows]

    def get_document_ids_in_timerange(self, start_date: datetime, end_date: datetime) -> list[int]:
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM documents
                WHERE CASE
                    WHEN file_mtime IS NOT NULL THEN file_mtime >= %s AND file_mtime < %s
                    ELSE updated_at >= %s AND updated_at < %s
                END
                """,
                (start_ts, end_ts, start_date, end_date),
            ).fetchall()
            return [row["id"] for row in rows]

    def get_chunk_public_fields(self, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch v0.14 public chunk fields keyed by internal ``chunks.id``."""
        if not chunk_ids:
            return {}
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, chunk_id, chunk_index, document_size,
                       source_created_at, chunk_source_uri
                FROM chunks WHERE id = ANY(%s)
                """,
                (list(chunk_ids),),
            ).fetchall()
        return {
            row["id"]: {
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "document_size": row["document_size"],
                "source_created_at": row["source_created_at"],
                "chunk_source_uri": row["chunk_source_uri"],
            }
            for row in rows
        }

    def get_stats(self) -> dict[str, Any]:
        with self._connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            embedding_count = conn.execute("SELECT COUNT(*) AS n FROM chunk_embeddings").fetchone()[
                "n"
            ]
            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "embedding_count": embedding_count,
                "database_path": self._sanitized_dsn(),
            }

    def _sanitized_dsn(self) -> str:
        """A display-safe DSN (password stripped) for stats/logging."""
        dsn = self.dsn or ""
        if "@" in dsn and "://" in dsn:
            scheme, rest = dsn.split("://", 1)
            if "@" in rest:
                creds, host = rest.split("@", 1)
                user = creds.split(":", 1)[0]
                return f"{scheme}://{user}:***@{host}"
        return dsn

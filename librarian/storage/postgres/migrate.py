"""
Postgres v0.14 schema migration.

Creates the librarian-owned schema on Postgres, mirroring the SQLite v0.14
shape (see :mod:`librarian.storage.migrate`). Differences are substrate-native,
not semantic:

* embeddings live in ``vector(dim)`` columns (pgvector) instead of sqlite-vec
  virtual tables;
* full-text search uses a ``tsvector`` generated column + GIN index instead of
  an FTS5 external-content table with sync triggers (the generated column keeps
  the index in lock-step with ``chunks.content`` automatically);
* ``cursor``/``metadata`` are ``JSONB``.

Every statement is ``IF NOT EXISTS`` / ``ADD COLUMN IF NOT EXISTS``, so
``migrate()`` is idempotent: re-running it on an up-to-date database is a no-op.
"""

import logging
from typing import Any

from librarian.config import (
    CODE_EMBEDDING_DIMENSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_DIMENSION,
    VISION_EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

V14_SCHEMA_VERSION = 14


def effective_text_dimension() -> int:
    """Text embedding dimension for the configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_EMBEDDING_DIMENSION
    return EMBEDDING_DIMENSION


def migrate(conn: Any, schema: str = "public") -> None:
    """Bring ``conn``'s database to the v0.14 Postgres schema (idempotent).

    Args:
        conn: An open psycopg connection.
        schema: Schema to create the librarian tables in.
    """
    from psycopg import sql

    text_dim = effective_text_dimension()
    code_dim = CODE_EMBEDDING_DIMENSION
    vision_dim = VISION_EMBEDDING_DIMENSION

    logger.info("Running v0.14 Postgres schema migration (schema=%s)", schema)

    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        # Pin pgvector to ``public`` (always on the search_path) rather than the
        # current schema. Installing it into a per-deployment / per-test schema
        # would make the ``vector`` type vanish when that schema is dropped, even
        # though ``CREATE EXTENSION IF NOT EXISTS`` still considers it present.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now(),
                file_mtime DOUBLE PRECISION,
                asset_type TEXT,
                document_id TEXT,
                document_source_uri TEXT,
                source_created_at TEXT
            )
            """
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                heading_path TEXT,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                asset_type TEXT,
                modality TEXT,
                chunk_id TEXT,
                document_size INTEGER,
                source_created_at TEXT,
                deleted_at TEXT,
                deletion_reason TEXT,
                document_source_uri TEXT,
                chunk_source_uri TEXT,
                content_tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', content)) STORED
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deleted_at ON chunks(deleted_at)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv)"
        )

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding vector({text_dim}),
                model_version TEXT
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vec_chunks_code (
                chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding vector({code_dim})
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vec_chunks_vision (
                chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding vector({vision_dim})
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                source_key TEXT PRIMARY KEY,
                cursor JSONB,
                status TEXT DEFAULT 'idle',
                last_success_at TEXT,
                last_attempt_at TEXT,
                last_error TEXT,
                documents_seen INTEGER DEFAULT 0,
                chunks_written INTEGER DEFAULT 0,
                config_version INTEGER DEFAULT 0
            )
            """
        )

    conn.commit()
    logger.info("v0.14 Postgres schema migration complete")

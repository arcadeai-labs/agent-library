"""
SQLite database management for librarian.

This module handles database connection, schema initialization,
and provides the core Database class for all storage operations.
"""

import json
import logging
import sqlite3
import struct
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import sqlite_vec

from librarian.config import (
    DATABASE_PATH,
    EMBEDDING_DIMENSION,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_DIMENSION,
    ensure_directories,
)
from librarian.storage.migrations import run_migrations
from librarian.types import AssetType, Chunk, Document, EmbeddingModality

logger = logging.getLogger(__name__)


def get_effective_embedding_dimension() -> int:
    """Get the embedding dimension based on configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_EMBEDDING_DIMENSION
    return EMBEDDING_DIMENSION


# Re-export types for backward compatibility
__all__ = [
    "Chunk",
    "Database",
    "Document",
    "deserialize_embedding",
    "get_database",
    "get_effective_embedding_dimension",
    "serialize_embedding",
]


def serialize_embedding(embedding: list[float]) -> bytes:
    """
    Serialize a list of floats to bytes for sqlite-vec storage.

    Args:
        embedding: List of float values representing the embedding vector.

    Returns:
        Bytes representation of the embedding.
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    """
    Deserialize bytes to a list of floats from sqlite-vec storage.

    Args:
        data: Bytes representation of the embedding.

    Returns:
        List of float values.
    """
    num_floats = len(data) // 4
    return list(struct.unpack(f"{num_floats}f", data))


class Database:
    """
    SQLite database manager with sqlite-vec and FTS5 support.

    Handles connection management, schema initialization, and provides
    thread-safe access to the database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file. Uses config default if None.
        """
        self.db_path = db_path or DATABASE_PATH
        self._local = threading.local()
        self._lock = threading.RLock()

        ensure_directories()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # Load sqlite-vec extension
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.connection = conn
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Yields:
            A sqlite3 connection with sqlite-vec loaded.
        """
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _get_vector_dimension(self, conn: sqlite3.Connection) -> int | None:
        """
        Get the dimension of the existing vector table.

        Returns:
            Dimension of the vector table, or None if table doesn't exist.
        """
        try:
            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
            )
            if not cursor.fetchone():
                return None

            # Get table schema to extract dimension
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name='chunk_embeddings'")
            schema = cursor.fetchone()
            if schema:
                sql = schema[0]
                # Parse: embedding float[1024]
                import re

                match = re.search(r"float\[(\d+)\]", sql)
                if match:
                    return int(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to get vector dimension: {e}")
        return None

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self.connection() as conn:
            # Check for dimension mismatch before creating tables
            existing_dim = self._get_vector_dimension(conn)
            expected_dim = get_effective_embedding_dimension()

            if existing_dim is not None and existing_dim != expected_dim:
                error_msg = (
                    f"Database dimension mismatch: expected {expected_dim} ({EMBEDDING_PROVIDER}), "
                    f"but database has {existing_dim}. Remove or rename ~/.librarian/index.db to fix."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_mtime REAL
                )
            """)

            # Migration: add file_mtime column if missing (for existing databases)
            cursor = conn.execute("PRAGMA table_info(documents)")
            columns = {row[1] for row in cursor.fetchall()}
            if "file_mtime" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN file_mtime REAL")

            # Create chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    heading_path TEXT,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)")

            # Create sqlite-vec virtual table for vector search
            # Uses provider-appropriate dimension (1024 for OpenAI, 384 for local)
            # Uses cosine distance for normalized embeddings (GoEmbed normalizes by default)
            dim = get_effective_embedding_dimension()
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{dim}] distance_metric=cosine
                )
            """)

            # Create FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    content='chunks',
                    content_rowid='id'
                )
            """)

            # Create triggers to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
                END
            """)

            # Run migrations to add multi-modal support
            run_migrations(conn)

            logger.info("Database schema initialized at %s", self.db_path)

    # =========================================================================
    # Document Operations
    # =========================================================================

    def insert_document(self, document: Document) -> int:
        """
        Insert a new document into the database.

        Args:
            document: The document to insert.

        Returns:
            The ID of the inserted document.
        """
        with self._lock, self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (path, title, content, metadata, file_mtime, asset_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.path,
                    document.title,
                    document.content,
                    json.dumps(document.metadata) if document.metadata else None,
                    document.file_mtime,
                    document.asset_type.value,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def update_document(self, document: Document) -> None:
        """
        Update an existing document.

        Args:
            document: The document to update (must have an id).
        """
        with self._lock, self.connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET path = ?, title = ?, content = ?, metadata = ?,
                    file_mtime = ?, asset_type = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    document.path,
                    document.title,
                    document.content,
                    json.dumps(document.metadata) if document.metadata else None,
                    document.file_mtime,
                    document.asset_type.value,
                    document.id,
                ),
            )

    def get_document_by_path(self, path: str) -> Document | None:
        """
        Get a document by its file path.

        Args:
            path: The file path of the document.

        Returns:
            The document if found, None otherwise.
        """
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
            if row:
                asset_type = AssetType(row["asset_type"]) if row["asset_type"] else AssetType.TEXT
                return Document(
                    id=row["id"],
                    path=row["path"],
                    title=row["title"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    asset_type=asset_type,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    file_mtime=row["file_mtime"],
                )
            return None

    def get_document_by_id(self, doc_id: int) -> Document | None:
        """
        Get a document by its ID.

        Args:
            doc_id: The document ID.

        Returns:
            The document if found, None otherwise.
        """
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row:
                asset_type = AssetType(row["asset_type"]) if row["asset_type"] else AssetType.TEXT
                return Document(
                    id=row["id"],
                    path=row["path"],
                    title=row["title"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    asset_type=asset_type,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    file_mtime=row["file_mtime"],
                )
            return None

    def delete_document(self, doc_id: int) -> None:
        """
        Delete a document and all its chunks.

        Args:
            doc_id: The document ID to delete.
        """
        with self._lock, self.connection() as conn:
            # Delete embeddings from all modality tables
            for table in ["chunk_embeddings", "vec_chunks_code", "vec_chunks_vision"]:
                conn.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
                    """,  # noqa: S608
                    (doc_id,),
                )
            # Chunks and FTS entries are deleted via CASCADE and triggers
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def delete_document_by_path(self, path: str) -> bool:
        """
        Delete a document by its file path.

        Args:
            path: The file path of the document.

        Returns:
            True if a document was deleted, False otherwise.
        """
        doc = self.get_document_by_path(path)
        if doc and doc.id:
            self.delete_document(doc.id)
            return True
        return False

    def list_documents(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Document]:
        """
        List documents in the database, optionally filtered by date range.

        Args:
            start_date: Optional start date for filtering (inclusive).
            end_date: Optional end date for filtering (exclusive).

        Returns:
            List of documents matching the criteria.
        """
        with self.connection() as conn:
            if start_date and end_date:
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE updated_at >= ? AND updated_at < ?
                    ORDER BY updated_at DESC
                    """,
                    (start_date.isoformat(), end_date.isoformat()),
                ).fetchall()
            elif start_date:
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE updated_at >= ?
                    ORDER BY updated_at DESC
                    """,
                    (start_date.isoformat(),),
                ).fetchall()
            elif end_date:
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE updated_at < ?
                    ORDER BY updated_at DESC
                    """,
                    (end_date.isoformat(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()

            return [
                Document(
                    id=row["id"],
                    path=row["path"],
                    title=row["title"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    asset_type=AssetType(row["asset_type"])
                    if row["asset_type"]
                    else AssetType.TEXT,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    file_mtime=row["file_mtime"],
                )
                for row in rows
            ]

    def get_document_ids_in_timerange(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[int]:
        """
        Get document IDs that fall within a time range.

        Args:
            start_date: Start of the time range (inclusive).
            end_date: End of the time range (exclusive).

        Returns:
            List of document IDs.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM documents
                WHERE updated_at >= ? AND updated_at < ?
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
            return [row["id"] for row in rows]

    # =========================================================================
    # Chunk Operations
    # =========================================================================

    def insert_chunk(self, chunk: Chunk) -> int:
        """
        Insert a chunk into the database.

        Args:
            chunk: The chunk to insert.

        Returns:
            The ID of the inserted chunk.
        """
        with self._lock, self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chunks
                (document_id, content, heading_path, chunk_index, start_char, end_char,
                 asset_type, modality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.document_id,
                    chunk.content,
                    chunk.heading_path,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.asset_type.value,
                    chunk.modality.value,
                ),
            )
            chunk_id = cursor.lastrowid

            # Insert embedding if provided, into the appropriate modality-specific table
            if chunk.embedding:
                if chunk.modality == EmbeddingModality.CODE:
                    table = "vec_chunks_code"
                elif chunk.modality == EmbeddingModality.VISION:
                    table = "vec_chunks_vision"
                else:
                    table = "chunk_embeddings"

                conn.execute(
                    f"INSERT INTO {table} (chunk_id, embedding) VALUES (?, ?)",  # noqa: S608
                    (chunk_id, serialize_embedding(chunk.embedding)),
                )

            return chunk_id  # type: ignore[return-value]

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """
        Insert multiple chunks in a batch.

        Args:
            chunks: List of chunks to insert.

        Returns:
            List of inserted chunk IDs.
        """
        chunk_ids = []
        with self._lock, self.connection() as conn:
            for chunk in chunks:
                cursor = conn.execute(
                    """
                    INSERT INTO chunks
                    (document_id, content, heading_path, chunk_index, start_char, end_char,
                     asset_type, modality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.document_id,
                        chunk.content,
                        chunk.heading_path,
                        chunk.chunk_index,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.asset_type.value,
                        chunk.modality.value,
                    ),
                )
                chunk_id = cursor.lastrowid
                if chunk_id is not None:
                    chunk_ids.append(chunk_id)

                if chunk.embedding:
                    if chunk.modality == EmbeddingModality.CODE:
                        table = "vec_chunks_code"
                    elif chunk.modality == EmbeddingModality.VISION:
                        table = "vec_chunks_vision"
                    else:
                        table = "chunk_embeddings"

                    conn.execute(
                        f"INSERT INTO {table} (chunk_id, embedding) VALUES (?, ?)",  # noqa: S608
                        (chunk_id, serialize_embedding(chunk.embedding)),
                    )

        return chunk_ids

    def get_chunks_by_document(self, doc_id: int) -> list[Chunk]:
        """
        Get all chunks for a document.

        Args:
            doc_id: The document ID.

        Returns:
            List of chunks for the document.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*, ce.embedding as text_emb, cc.embedding as code_emb, cv.embedding as vision_emb
                FROM chunks c
                LEFT JOIN chunk_embeddings ce ON c.id = ce.chunk_id
                LEFT JOIN vec_chunks_code cc ON c.id = cc.chunk_id
                LEFT JOIN vec_chunks_vision cv ON c.id = cv.chunk_id
                WHERE c.document_id = ?
                ORDER BY c.chunk_index
                """,
                (doc_id,),
            ).fetchall()
            chunks = []
            for row in rows:
                asset_type = AssetType(row["asset_type"]) if row["asset_type"] else AssetType.TEXT
                modality = (
                    EmbeddingModality(row["modality"])
                    if row["modality"]
                    else EmbeddingModality.TEXT
                )

                # Get embedding from appropriate table
                embedding = None
                if row["text_emb"]:
                    embedding = deserialize_embedding(row["text_emb"])
                elif row["code_emb"]:
                    embedding = deserialize_embedding(row["code_emb"])
                elif row["vision_emb"]:
                    embedding = deserialize_embedding(row["vision_emb"])

                chunks.append(
                    Chunk(
                        id=row["id"],
                        document_id=row["document_id"],
                        content=row["content"],
                        heading_path=row["heading_path"],
                        chunk_index=row["chunk_index"],
                        start_char=row["start_char"],
                        end_char=row["end_char"],
                        embedding=embedding,
                        asset_type=asset_type,
                        modality=modality,
                    )
                )
            return chunks

    def delete_chunks_by_document(self, doc_id: int) -> None:
        """
        Delete all chunks for a document.

        Args:
            doc_id: The document ID.
        """
        with self._lock, self.connection() as conn:
            # Delete from all embedding tables
            for table in ["chunk_embeddings", "vec_chunks_code", "vec_chunks_vision"]:
                conn.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
                    """,  # noqa: S608
                    (doc_id,),
                )
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics about documents and chunks.
        """
        with self.connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedding_count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]

            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "embedding_count": embedding_count,
                "database_path": self.db_path,
            }

    def clear_all(self) -> None:
        """
        Delete all data from the database.

        Removes all documents, chunks, embeddings, and FTS data.
        Use for rebuilding the index from scratch.
        """
        import contextlib

        with self._lock, self.connection() as conn:
            # Delete in order respecting foreign keys, ignore missing tables
            for table in ["chunk_fts", "chunk_embeddings", "chunks", "documents"]:
                with contextlib.suppress(Exception):
                    conn.execute(f"DELETE FROM {table}")  # noqa: S608
            conn.commit()


# Global database instance
_db_instance: Database | None = None
_db_lock = threading.Lock()


def get_database() -> Database:
    """
    Get the global database instance.

    Returns:
        The global Database instance.
    """
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = Database()
    return _db_instance

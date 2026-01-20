"""
Indexing service for document management.

Coordinates the full indexing pipeline: processing files and storing
them in the database with embeddings.
"""

import logging
from pathlib import Path
from typing import Any

from librarian.processing.embed import get_embedder
from librarian.processing.parsers.md import MarkdownParser
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy
from librarian.storage.database import get_database
from librarian.types import Chunk, Document

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service for indexing documents into the database.

    Coordinates parsing, chunking, embedding generation, and database storage.
    """

    def __init__(self) -> None:
        """Initialize the indexing service with default components."""
        self._parser = MarkdownParser()
        self._chunker = Chunker(strategy=ChunkingStrategy.HEADERS)

    def index_file(self, file_path: Path, timeout: float = 5.0) -> dict[str, Any]:
        """
        Process and index a markdown file.

        Args:
            file_path: Path to the markdown file.
            timeout: Max seconds to wait for file read (for network filesystems).

        Returns:
            Dictionary with indexing results including path, title, chunk count, status.

        Raises:
            TimeoutError: If file read times out (e.g., iCloud not synced).
            FileNotFoundError: If file doesn't exist.
        """
        db = get_database()
        embedder = get_embedder()

        # Get file modification time for change detection
        try:
            file_mtime = file_path.stat().st_mtime
        except (OSError, TimeoutError) as e:
            # Re-raise with context for network/cloud filesystem issues
            raise TimeoutError(str(file_path)) from e

        # Parse the document
        parsed = self._parser.parse_file(file_path)

        # Check if document exists for update vs insert
        existing = db.get_document_by_path(str(file_path))
        if existing and existing.id:
            # Update existing document
            db.delete_chunks_by_document(existing.id)
            existing.title = parsed.title
            existing.content = parsed.content
            existing.metadata = parsed.metadata
            existing.file_mtime = file_mtime
            db.update_document(existing)
            doc_id = existing.id
            status = "updated"
        else:
            # Insert new document
            doc = Document(
                id=None,
                path=str(file_path),
                title=parsed.title,
                content=parsed.content,
                metadata=parsed.metadata,
                file_mtime=file_mtime,
            )
            doc_id = db.insert_document(doc)
            status = "created"

        # Chunk and embed (use embed_documents for proper instruction-based embedding)
        chunks = self._chunker.chunk_document(parsed)
        chunk_texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(chunk_texts)

        # Store chunks with embeddings
        db_chunks = [
            Chunk(
                id=None,
                document_id=doc_id,
                content=chunk.content,
                heading_path=chunk.heading_path,
                chunk_index=i,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding=embedding,
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        db.insert_chunks_batch(db_chunks)

        return {
            "path": str(file_path),
            "title": parsed.title,
            "chunks": len(chunks),
            "status": status,
        }

    def should_reindex(self, file_path: Path) -> bool:
        """
        Check if a file needs reindexing based on modification time.

        Args:
            file_path: Path to check.

        Returns:
            True if file should be reindexed, False if unchanged.
        """
        db = get_database()
        current_mtime = file_path.stat().st_mtime

        existing = db.get_document_by_path(str(file_path))
        if not existing:
            return True

        if existing.file_mtime is None:
            return True

        return current_mtime > existing.file_mtime


# Global singleton
_indexing_service: IndexingService | None = None


def get_indexing_service() -> IndexingService:
    """Get the global indexing service instance."""
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = IndexingService()
    return _indexing_service

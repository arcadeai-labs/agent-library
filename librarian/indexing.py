"""
Indexing service for document management.

Coordinates the full indexing pipeline: processing files and storing
them in the database with embeddings.
"""

import logging
from pathlib import Path
from typing import Any

from librarian.processing.embed import get_embedder
from librarian.processing.parsers.registry import get_parser_for_file
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy
from librarian.processing.transform.code import CodeChunker, chunk_code_by_blocks
from librarian.processing.transform.pdf import PDFChunker
from librarian.storage.database import get_database
from librarian.types import AssetType, Chunk, Document, EmbeddingModality

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service for indexing documents into the database.

    Coordinates parsing, chunking, embedding generation, and database storage.
    """

    def __init__(self) -> None:
        """Initialize the indexing service with default components."""
        self._text_chunker = Chunker(strategy=ChunkingStrategy.HEADERS)
        self._code_chunker = CodeChunker()
        self._pdf_chunker = PDFChunker()

    def index_file(self, file_path: Path, timeout: float = 5.0) -> dict[str, Any]:
        """
        Process and index a file (text, code, PDF, image).

        Args:
            file_path: Path to the file.
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

        # Get appropriate parser from registry
        parser, asset_type = get_parser_for_file(file_path)
        if parser is None:
            logger.warning(f"No parser found for {file_path}, skipping")
            return {
                "path": str(file_path),
                "title": None,
                "chunks": 0,
                "status": "skipped",
                "reason": "no parser found",
            }

        # Parse the document
        parsed = parser.parse_file(file_path)

        # Check if document exists for update vs insert
        existing = db.get_document_by_path(str(file_path))
        if existing and existing.id:
            # Update existing document
            db.delete_chunks_by_document(existing.id)
            existing.title = parsed.title
            existing.content = parsed.content
            existing.metadata = parsed.metadata
            existing.file_mtime = file_mtime
            existing.asset_type = asset_type
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
                asset_type=asset_type,
            )
            doc_id = db.insert_document(doc)
            status = "created"

        # Chunk based on asset type
        if asset_type == AssetType.CODE:
            # Use code-aware chunking
            symbols = parsed.metadata.get("symbols", [])
            if symbols:
                # We have symbols from the parser, convert back to CodeSymbol objects
                from librarian.types import CodeSymbol, CodeSymbolType

                code_symbols = [
                    CodeSymbol(
                        name=s["name"],
                        symbol_type=CodeSymbolType(s["type"]),
                        line_start=s["line_start"],
                        line_end=s["line_end"],
                    )
                    for s in symbols
                ]
                chunks = self._code_chunker.chunk_by_symbols(
                    parsed.content, code_symbols, parsed.metadata
                )
            else:
                # No symbols, use block-based chunking
                language = parsed.metadata.get("language", "unknown")
                chunks = chunk_code_by_blocks(parsed.content, language, parsed.metadata)
        elif asset_type == AssetType.PDF:
            # Use PDF chunking by pages
            page_count = parsed.metadata.get("page_count", 1)
            chunks = self._pdf_chunker.chunk_by_pages(parsed.content, page_count, parsed.metadata)
        elif asset_type == AssetType.IMAGE:
            # Images are single chunks (no chunking needed)
            from librarian.types import TextChunk

            chunks = [
                TextChunk(
                    content=parsed.content,
                    index=0,
                    start_char=0,
                    end_char=len(parsed.content),
                    heading_path=parsed.title,
                    metadata=parsed.metadata,
                )
            ]
        else:
            # Use text chunker for TEXT and other types
            chunks = self._text_chunker.chunk_document(parsed)

        # Embed chunks
        chunk_texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(chunk_texts)

        # Determine embedding modality
        # For now, we use TEXT modality for all assets since we're using text embedding models
        # TODO: When specialized models are available (CodeBERT, CLIP), use:
        #   - EmbeddingModality.CODE for code with CodeBERT
        #   - EmbeddingModality.VISION for images with CLIP
        #   - EmbeddingModality.TEXT for text documents
        modality = EmbeddingModality.TEXT

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
                asset_type=asset_type,
                modality=modality,
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

"""
Indexing service for document management.

Coordinates the full indexing pipeline: processing files and storing
them in the database with embeddings. Supports multi-modal embeddings
for text, code, and vision content.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from librarian.config import ENABLE_CODE_EMBEDDINGS, ENABLE_VISION_EMBEDDINGS
from librarian.processing.embed import get_embedder, get_embedder_for_modality
from librarian.processing.parsers.base import FileReadError, FileReadTimeoutError
from librarian.processing.parsers.registry import get_parser_for_file
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy
from librarian.processing.transform.code import CodeChunker, chunk_code_by_blocks
from librarian.processing.transform.pdf import PDFChunker
from librarian.storage.database import get_database
from librarian.types import AssetType, Chunk, Document, EmbeddingModality

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service for indexing documents into the database.

    Coordinates parsing, chunking, embedding generation, and database storage.
    Supports multi-modal embeddings (text, code, vision) when enabled.
    """

    def __init__(self) -> None:
        """Initialize the indexing service with default components."""
        self._text_chunker = Chunker(strategy=ChunkingStrategy.HEADERS)
        self._code_chunker = CodeChunker()
        self._pdf_chunker = PDFChunker()

    def _determine_modality(self, asset_type: AssetType) -> EmbeddingModality:
        """
        Determine embedding modality based on asset type and configuration.

        Uses specialized embedding models (CodeBERT, CLIP) when enabled,
        otherwise falls back to text embeddings.

        Args:
            asset_type: The type of asset being indexed.

        Returns:
            The appropriate embedding modality for the asset.
        """
        if asset_type == AssetType.CODE and ENABLE_CODE_EMBEDDINGS:
            return EmbeddingModality.CODE
        elif asset_type == AssetType.IMAGE and ENABLE_VISION_EMBEDDINGS:
            return EmbeddingModality.VISION
        return EmbeddingModality.TEXT

    def _load_image(self, file_path: Path) -> "PILImage | None":  # type: ignore[no-any-unimported]
        """
        Load a PIL Image from a file path.

        Args:
            file_path: Path to the image file.

        Returns:
            PIL Image object, or None if loading fails.
        """
        try:
            from PIL import Image

            return Image.open(file_path)
        except ImportError:
            logger.warning("PIL not available for image loading")
            return None
        except Exception as e:
            logger.warning(f"Failed to load image {file_path}: {e}")
            return None

    def _embed_image_chunks(
        self, file_path: Path, chunks: list[Any], embedder: Any
    ) -> list[list[float]]:
        """
        Embed image chunks using actual PIL Image data.

        For images with vision embeddings enabled, this loads the actual
        image and passes it to the CLIP model for true visual embedding.

        Args:
            file_path: Path to the image file.
            chunks: List of chunks (for images, usually just one).
            embedder: The vision embedder (CLIP-based).

        Returns:
            List of embedding vectors for each chunk.
        """
        img = self._load_image(file_path)
        if img is None:
            # Fall back to text embedding of image description
            logger.warning(f"Could not load image {file_path}, using text description")
            result: list[list[float]] = embedder.embed_documents([c.content for c in chunks])
            return result

        # For images, we typically have a single chunk representing the whole image
        # We embed the actual image pixels, not the text description
        try:
            # Use embed_image method if available (LocalEmbeddingProvider)
            if hasattr(embedder, "embed_image"):
                embedding: list[float] = embedder.embed_image(img)
                img.close()
                # Return same embedding for all chunks (usually just one for images)
                return [embedding] * len(chunks)
            else:
                # For embedders that support images in embed() directly
                embedding = embedder.embed(img)
                img.close()
                return [embedding] * len(chunks)
        except Exception as e:
            logger.warning(f"Vision embedding failed for {file_path}: {e}, using text fallback")
            img.close()
            fallback: list[list[float]] = embedder.embed_documents([c.content for c in chunks])
            return fallback

    def index_file(self, file_path: Path) -> dict[str, Any]:
        """
        Process and index a file (text, code, PDF, image).

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary with indexing results including path, title, chunk count, status.

        Raises:
            FileReadTimeoutError: If file read times out (e.g., iCloud not synced).
            FileReadError: For I/O errors (permissions, etc.).
            FileNotFoundError: If file doesn't exist.
        """
        db = get_database()

        # Get file modification time for change detection
        try:
            file_mtime = file_path.stat().st_mtime
        except TimeoutError as e:
            raise FileReadTimeoutError(
                f"Timed out accessing {file_path} (file may not be synced from cloud storage)"
            ) from e
        except OSError as e:
            raise FileReadError(f"Cannot access {file_path}: {e}") from e

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

        # Parse the document (parsers handle their own timeout/IO errors)
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

        # Determine embedding modality and get appropriate embedder
        modality = self._determine_modality(asset_type)
        embedder = get_embedder_for_modality(modality)

        # Fall back to text embedder if specialized one unavailable
        if embedder is None:
            logger.debug(f"Modality {modality.value} embedder not available, falling back to TEXT")
            modality = EmbeddingModality.TEXT
            embedder = get_embedder()

        # Embed chunks - use PIL Image for vision embeddings
        if modality == EmbeddingModality.VISION and asset_type == AssetType.IMAGE:
            embeddings = self._embed_image_chunks(file_path, chunks, embedder)
        else:
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

        Raises:
            FileReadTimeoutError: If stat() times out.
            FileReadError: For I/O errors.
        """
        db = get_database()

        try:
            current_mtime = file_path.stat().st_mtime
        except TimeoutError as e:
            raise FileReadTimeoutError(
                f"Timed out accessing {file_path} (file may not be synced from cloud storage)"
            ) from e
        except OSError as e:
            raise FileReadError(f"Cannot access {file_path}: {e}") from e

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

"""
Processing manager for document ingestion pipeline.

Coordinates the flow from raw files through parsing, chunking, and embedding
to produce index-ready data.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from librarian.processing.embed import Embedder, get_embedder
from librarian.processing.parsers.base import BaseParser
from librarian.processing.parsers.md import MarkdownParser
from librarian.processing.parsers.obsidian import ObsidianParser
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy
from librarian.types import ParsedDocument, TextChunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ProcessingManager:
    """
    Coordinates the document processing pipeline.

    Handles:
    1. Parser selection based on file type
    2. Document parsing
    3. Text chunking
    4. Embedding generation

    Thread-safe for concurrent processing.
    """

    def __init__(
        self,
        parser: BaseParser | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        auto_detect_parser: bool = True,
    ) -> None:
        """
        Initialize the processing manager.

        Args:
            parser: Parser to use (default: MarkdownParser).
            chunker: Chunker to use (default: header-based chunker).
            embedder: Embedder to use (default: from config).
            auto_detect_parser: If True, select parser based on file content.
        """
        self._default_parser = parser or MarkdownParser()
        self._chunker = chunker or Chunker(strategy=ChunkingStrategy.HEADERS)
        self._embedder = embedder or get_embedder()
        self._auto_detect = auto_detect_parser

        # Parser registry for auto-detection
        self._parsers: dict[str, BaseParser] = {
            "markdown": MarkdownParser(),
            "obsidian": ObsidianParser(),
        }

    @property
    def embedder(self) -> Embedder:
        """Return the embedder instance."""
        return self._embedder

    @property
    def chunker(self) -> Chunker:
        """Return the chunker instance."""
        return self._chunker

    def process_file(
        self, file_path: Path | str
    ) -> tuple[ParsedDocument, list[TextChunk], list[list[float]]]:
        """
        Process a single file through the full pipeline.

        Args:
            file_path: Path to the file to process.

        Returns:
            Tuple of (parsed_document, chunks, embeddings).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file cannot be parsed.
        """
        path = Path(file_path)
        parser = self._select_parser(path)

        # Parse
        logger.debug("Parsing %s with %s", path, type(parser).__name__)
        document = parser.parse_file(path)

        # Chunk
        chunks = self._chunker.chunk_document(document)
        logger.debug("Created %d chunks from %s", len(chunks), path)

        # Embed (use embed_documents for proper instruction-based embedding)
        chunk_texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_documents(chunk_texts)

        return document, chunks, embeddings

    def process_content(
        self,
        content: str,
        path: str = "",
        parser_type: str | None = None,
    ) -> tuple[ParsedDocument, list[TextChunk], list[list[float]]]:
        """
        Process content string through the pipeline.

        Args:
            content: Document content as string.
            path: Optional path for reference.
            parser_type: Parser to use ("markdown" or "obsidian").

        Returns:
            Tuple of (parsed_document, chunks, embeddings).
        """
        parser = self._parsers.get(parser_type or "markdown", self._default_parser)

        document = parser.parse_content(content, path)
        chunks = self._chunker.chunk_document(document)

        chunk_texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_documents(chunk_texts)

        return document, chunks, embeddings

    def process_batch(
        self, file_paths: list[Path | str]
    ) -> list[tuple[ParsedDocument, list[TextChunk], list[list[float]]]]:
        """
        Process multiple files.

        Optimizes embedding by batching across all documents.

        Args:
            file_paths: List of file paths to process.

        Returns:
            List of (document, chunks, embeddings) tuples.
        """
        # Parse and chunk all documents first
        all_docs: list[tuple[ParsedDocument, list[TextChunk]]] = []
        all_chunk_texts: list[str] = []
        chunk_counts: list[int] = []

        for path in file_paths:
            try:
                path = Path(path)
                parser = self._select_parser(path)
                document = parser.parse_file(path)
                chunks = self._chunker.chunk_document(document)

                all_docs.append((document, chunks))
                all_chunk_texts.extend(c.content for c in chunks)
                chunk_counts.append(len(chunks))

            except Exception:
                logger.exception("Failed to process %s", path)
                continue

        # Batch embed all chunks at once (use embed_documents for instruction-based embedding)
        logger.info("Embedding %d chunks from %d documents", len(all_chunk_texts), len(all_docs))
        all_embeddings = self._embedder.embed_documents(all_chunk_texts)

        # Split embeddings back to per-document
        results: list[tuple[ParsedDocument, list[TextChunk], list[list[float]]]] = []
        offset = 0

        for (doc, chunks), count in zip(all_docs, chunk_counts, strict=True):
            doc_embeddings = all_embeddings[offset : offset + count]
            results.append((doc, chunks, doc_embeddings))
            offset += count

        return results

    def _select_parser(self, file_path: Path) -> BaseParser:
        """
        Select appropriate parser for a file.

        Detection heuristics:
        1. Check file extension
        2. Check for Obsidian-specific content (wiki-links, callouts)
        """
        if not self._auto_detect:
            return self._default_parser

        # Check extension
        suffix = file_path.suffix.lower()
        if suffix not in {".md", ".markdown", ".mdown", ".mkd"}:
            return self._default_parser

        # Quick content check for Obsidian features
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")[:2000]

            # Look for Obsidian-specific syntax
            has_wikilinks = "[[" in content and "]]" in content
            has_callouts = ">[!" in content or "> [!" in content

            if has_wikilinks or has_callouts:
                return self._parsers["obsidian"]

        except (OSError, UnicodeDecodeError):
            # File read errors are expected for non-text files
            pass

        return self._default_parser

    def parse_only(self, file_path: Path | str) -> ParsedDocument:
        """
        Parse a file without chunking or embedding.

        Useful for inspection or when only metadata is needed.
        """
        path = Path(file_path)
        parser = self._select_parser(path)
        return parser.parse_file(path)

    def chunk_only(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Chunk an already-parsed document.

        Useful when re-chunking with different settings.
        """
        return self._chunker.chunk_document(document)

    def embed_only(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for arbitrary text.

        Useful for query embedding or re-embedding.
        """
        return self._embedder.embed_batch(texts)

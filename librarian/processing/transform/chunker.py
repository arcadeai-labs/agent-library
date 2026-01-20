"""
Text chunking with configurable strategies.

This module provides functionality to split documents into chunks
with support for different chunking strategies and overlap.
"""

import logging
import re
from enum import Enum

from librarian.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE
from librarian.types import ParsedDocument, Section, TextChunk

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    HEADERS = "headers"  # Split by markdown headers
    PARAGRAPHS = "paragraphs"  # Split by paragraphs
    SENTENCES = "sentences"  # Split by sentences
    FIXED = "fixed"  # Fixed character/token count


class Chunker:
    """
    Text chunker with multiple strategies and overlap support.

    Supports chunking by headers, paragraphs, sentences, or fixed size,
    with configurable overlap between chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.HEADERS,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk (in characters).
            chunk_overlap: Number of characters to overlap between chunks.
            min_chunk_size: Minimum chunk size (smaller chunks are merged).
            strategy: The chunking strategy to use.
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or MIN_CHUNK_SIZE
        self.strategy = strategy

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Chunk a parsed document.

        Args:
            document: The parsed document to chunk.

        Returns:
            List of text chunks.
        """
        if self.strategy == ChunkingStrategy.HEADERS:
            return self._chunk_by_headers(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPHS:
            return self._chunk_by_paragraphs(document.content)
        elif self.strategy == ChunkingStrategy.SENTENCES:
            return self._chunk_by_sentences(document.content)
        else:
            return self._chunk_fixed(document.content)

    def chunk_text(self, text: str) -> list[TextChunk]:
        """
        Chunk raw text content.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        if self.strategy == ChunkingStrategy.PARAGRAPHS:
            return self._chunk_by_paragraphs(text)
        elif self.strategy == ChunkingStrategy.SENTENCES:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_fixed(text)

    def _chunk_by_headers(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Chunk document by markdown headers.

        Creates chunks for each section, with subsections included
        in parent sections if they're small enough.

        Args:
            document: The parsed document.

        Returns:
            List of chunks organized by headers.
        """
        chunks: list[TextChunk] = []
        chunk_index = 0

        if not document.sections:
            # No sections, chunk the entire content
            return self._chunk_fixed(document.content)

        for section in document.sections:
            section_chunks = self._chunk_section(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _chunk_section(self, section: Section, start_index: int) -> list[TextChunk]:
        """
        Chunk a single section.

        Args:
            section: The section to chunk.
            start_index: Starting chunk index.

        Returns:
            List of chunks from this section.
        """
        # Build full section text including header
        header_prefix = "#" * section.level + " " if section.level > 0 else ""
        full_text = f"{header_prefix}{section.title}\n\n{section.content}".strip()

        if len(full_text) <= self.chunk_size:
            # Section fits in one chunk
            return [
                TextChunk(
                    content=full_text,
                    index=start_index,
                    start_char=section.start_pos,
                    end_char=section.end_pos,
                    heading_path=section.title,
                )
            ]

        # Section too large, split with overlap
        sub_chunks = self._split_with_overlap(full_text, section.start_pos, section.title)
        for i, chunk in enumerate(sub_chunks):
            chunk.index = start_index + i

        return sub_chunks

    def _chunk_by_paragraphs(self, text: str) -> list[TextChunk]:
        """
        Chunk text by paragraphs.

        Args:
            text: The text to chunk.

        Returns:
            List of paragraph-based chunks.
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[TextChunk] = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        pos = 0

        for para in paragraphs:
            para_start = text.find(para, pos)
            para_end = para_start + len(para)
            pos = para_end

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = para_start
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(
                        TextChunk(
                            content=current_chunk,
                            index=chunk_index,
                            start_char=current_start,
                            end_char=para_start,
                        )
                    )
                    chunk_index += 1

                # Start new chunk
                current_chunk = para
                current_start = para_start

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                TextChunk(
                    content=current_chunk,
                    index=chunk_index,
                    start_char=current_start,
                    end_char=len(text),
                )
            )

        return self._merge_small_chunks(chunks)

    def _chunk_by_sentences(self, text: str) -> list[TextChunk]:
        """
        Chunk text by sentences.

        Args:
            text: The text to chunk.

        Returns:
            List of sentence-based chunks.
        """
        # Simple sentence splitting (handles common cases)
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        chunks: list[TextChunk] = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_start = text.find(sentence, pos)
            if sent_start == -1:
                sent_start = pos
            sent_end = sent_start + len(sentence)
            pos = sent_end

            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = sent_start
            else:
                if current_chunk:
                    chunks.append(
                        TextChunk(
                            content=current_chunk,
                            index=chunk_index,
                            start_char=current_start,
                            end_char=sent_start,
                        )
                    )
                    chunk_index += 1

                current_chunk = sentence
                current_start = sent_start

        if current_chunk:
            chunks.append(
                TextChunk(
                    content=current_chunk,
                    index=chunk_index,
                    start_char=current_start,
                    end_char=len(text),
                )
            )

        return self._merge_small_chunks(chunks)

    def _chunk_fixed(self, text: str) -> list[TextChunk]:
        """
        Chunk text at fixed character intervals with overlap.

        Args:
            text: The text to chunk.

        Returns:
            List of fixed-size chunks.
        """
        if len(text) <= self.chunk_size:
            return [
                TextChunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        return self._split_with_overlap(text, 0, None)

    def _split_with_overlap(
        self, text: str, base_start: int, heading_path: str | None
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to split.
            base_start: Base character position for start_char calculation.
            heading_path: Heading path for the chunks.

        Returns:
            List of overlapping chunks.
        """
        chunks: list[TextChunk] = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a natural boundary (space, newline)
            if end < len(text):
                # Look for a good break point
                break_chars = ["\n\n", "\n", ". ", " "]
                for break_char in break_chars:
                    break_pos = text.rfind(break_char, start + self.min_chunk_size, end)
                    if break_pos > start:
                        end = break_pos + len(break_char)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        index=chunk_index,
                        start_char=base_start + start,
                        end_char=base_start + end,
                        heading_path=heading_path,
                    )
                )
                chunk_index += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.min_chunk_size:
                break

        return chunks

    def _merge_small_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """
        Merge chunks that are too small.

        Args:
            chunks: List of chunks to process.

        Returns:
            List of chunks with small ones merged.
        """
        if not chunks:
            return chunks

        merged: list[TextChunk] = []
        current = chunks[0]

        for chunk in chunks[1:]:
            combined_len = len(current.content) + len(chunk.content) + 2

            if len(current.content) < self.min_chunk_size and combined_len <= self.chunk_size:
                # Merge with next chunk
                current = TextChunk(
                    content=current.content + "\n\n" + chunk.content,
                    index=current.index,
                    start_char=current.start_char,
                    end_char=chunk.end_char,
                    heading_path=current.heading_path or chunk.heading_path,
                )
            else:
                merged.append(current)
                current = chunk

        merged.append(current)

        # Re-index
        for i, chunk in enumerate(merged):
            chunk.index = i

        return merged

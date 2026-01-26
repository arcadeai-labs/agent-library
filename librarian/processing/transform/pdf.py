"""
PDF-aware text chunking.

Chunks PDF documents by pages or by fixed size with page tracking.
"""

from typing import Any

from librarian.types import TextChunk


class PDFChunker:
    """Chunker for PDF documents that preserves page structure."""

    def chunk_by_pages(
        self,
        content: str,
        page_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """
        Chunk PDF by pages.

        Assumes pages are separated by double newlines in the content.

        Args:
            content: Full PDF text content.
            page_count: Number of pages in the PDF.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of text chunks, one per page.
        """
        # Split content by page breaks (double newlines)
        pages = content.split("\n\n")

        chunks = []
        current_pos = 0

        for i in range(min(len(pages), page_count)):
            page_content = pages[i] if i < len(pages) else ""
            page_content = page_content.strip()

            if not page_content:
                current_pos += 2  # Account for the newlines
                continue

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["page_number"] = i + 1

            chunks.append(
                TextChunk(
                    content=page_content,
                    index=i,
                    start_char=current_pos,
                    end_char=current_pos + len(page_content),
                    heading_path=f"Page {i + 1}",
                    metadata=chunk_metadata,
                )
            )

            current_pos += len(page_content) + 2  # Account for separator

        return chunks

    def chunk_fixed_size(
        self,
        content: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """
        Chunk PDF by fixed character size with overlap.

        Args:
            content: Full PDF text content.
            chunk_size: Target chunk size in characters.
            overlap: Number of overlapping characters between chunks.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of text chunks.
        """
        chunks = []
        idx = 0
        pos = 0

        while pos < len(content):
            # Get chunk
            chunk_end = min(pos + chunk_size, len(content))
            chunk_content = content[pos:chunk_end]

            chunks.append(
                TextChunk(
                    content=chunk_content,
                    index=idx,
                    start_char=pos,
                    end_char=chunk_end,
                    metadata=metadata,
                )
            )

            # Move position with overlap
            pos = chunk_end - overlap
            if pos >= len(content) - overlap:
                break
            idx += 1

        return chunks

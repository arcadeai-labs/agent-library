"""
Code-aware text chunking.

Chunks source code by functions, classes, and other logical units
while preserving context.
"""

import re
from itertools import pairwise
from typing import Any

from librarian.config import CODE_CONTEXT_LINES, CODE_INCLUDE_CONTEXT
from librarian.types import CodeSymbol, TextChunk


class CodeChunker:
    """Chunker for source code that preserves logical structure."""

    def __init__(
        self,
        include_context: bool = CODE_INCLUDE_CONTEXT,
        context_lines: int = CODE_CONTEXT_LINES,
    ) -> None:
        """
        Initialize the code chunker.

        Args:
            include_context: Whether to include surrounding context lines.
            context_lines: Number of context lines to include before/after.
        """
        self.include_context = include_context
        self.context_lines = context_lines

    def chunk_by_symbols(
        self,
        content: str,
        symbols: list[CodeSymbol],
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """
        Chunk code by extracted symbols (functions, classes, etc.).

        Args:
            content: Full source code content.
            symbols: List of extracted code symbols.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of text chunks, one per symbol.
        """
        if not symbols:
            # No symbols found, return whole file as single chunk
            return [
                TextChunk(
                    content=content,
                    index=0,
                    start_char=0,
                    end_char=len(content),
                    metadata=metadata,
                )
            ]

        chunks = []
        lines = content.split("\n")

        for idx, symbol in enumerate(symbols):
            # Calculate line ranges with context
            if self.include_context:
                start_line = max(0, symbol.line_start - 1 - self.context_lines)
                end_line = min(len(lines), symbol.line_end + self.context_lines)
            else:
                start_line = symbol.line_start - 1
                end_line = symbol.line_end

            # Extract chunk content
            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines)

            # Calculate character positions
            start_char = sum(len(line) + 1 for line in lines[:start_line])
            end_char = start_char + len(chunk_content)

            # Create heading path
            heading = f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name

            # Add symbol-specific metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "symbol_name": symbol.name,
                "symbol_type": symbol.symbol_type.value,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
            })

            chunks.append(
                TextChunk(
                    content=chunk_content,
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    heading_path=heading,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def chunk_by_lines(
        self,
        content: str,
        lines_per_chunk: int = 50,
        overlap: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """
        Chunk code by fixed number of lines with overlap.

        Args:
            content: Source code content.
            lines_per_chunk: Number of lines per chunk.
            overlap: Number of overlapping lines between chunks.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of text chunks.
        """
        lines = content.split("\n")
        chunks = []
        idx = 0
        line_num = 0

        while line_num < len(lines):
            # Get chunk lines
            chunk_end = min(line_num + lines_per_chunk, len(lines))
            chunk_lines = lines[line_num:chunk_end]
            chunk_content = "\n".join(chunk_lines)

            # Calculate character positions
            start_char = sum(len(line) + 1 for line in lines[:line_num])
            end_char = start_char + len(chunk_content)

            chunks.append(
                TextChunk(
                    content=chunk_content,
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    heading_path=f"Lines {line_num + 1}-{chunk_end}",
                    metadata=metadata,
                )
            )

            # Move to next chunk with overlap
            line_num = chunk_end - overlap
            if line_num >= len(lines) - overlap:
                break
            idx += 1

        return chunks


def chunk_code_by_blocks(
    content: str,
    language: str,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """
    Chunk code by logical blocks (functions, classes, etc.).

    This is a convenience function that attempts to find natural code boundaries.

    Args:
        content: Source code content.
        language: Programming language.
        metadata: Optional metadata to attach to chunks.

    Returns:
        List of text chunks.
    """
    # Try to find function/class boundaries
    chunks = []
    lines = content.split("\n")

    # Patterns for common code block starts
    block_patterns = {
        "python": r"^\s*(def|class)\s+\w+",
        "javascript": r"^\s*(function|class|const\s+\w+\s*=\s*\()",
        "typescript": r"^\s*(function|class|interface|type|const\s+\w+\s*=)",
        "go": r"^\s*(func|type)\s+",
        "java": r"^\s*(public|private|protected)?\s*(class|interface|void|static)",
        "default": r"^\s*(function|def|class|fn|func)\s+",
    }

    pattern_str = block_patterns.get(language.lower(), block_patterns["default"])
    pattern = re.compile(pattern_str)

    block_starts = [0]  # Always start with beginning of file
    for i, line in enumerate(lines):
        if pattern.match(line):
            block_starts.append(i)
    block_starts.append(len(lines))  # Add end

    # Create chunks from blocks
    for idx, (start, end) in enumerate(pairwise(block_starts)):
        if end - start < 2:  # Skip tiny blocks
            continue

        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        # Calculate character positions
        start_char = sum(len(line) + 1 for line in lines[:start])
        end_char = start_char + len(chunk_content)

        # Try to extract block name
        first_line = chunk_lines[0] if chunk_lines else ""
        heading = first_line.strip()[:50]  # First 50 chars of defining line

        chunks.append(
            TextChunk(
                content=chunk_content,
                index=idx,
                start_char=start_char,
                end_char=end_char,
                heading_path=heading,
                metadata=metadata,
            )
        )

    # If no blocks found, return whole file
    if not chunks:
        return [
            TextChunk(
                content=content,
                index=0,
                start_char=0,
                end_char=len(content),
                metadata=metadata,
            )
        ]

    return chunks

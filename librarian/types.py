"""
Core types for librarian.

This module defines all internal representations used throughout the system:
- Ingested types: ParsedDocument, Section (from parsers)
- Computed types: TextChunk (from transform pipeline)
- Storage types: Document, Chunk (database records)
- Retrieval types: SearchResult (search output)

All modules should import types from here to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# =============================================================================
# Ingested Types (from parsers)
# =============================================================================


@dataclass
class Section:
    """
    Represents a section of a markdown document.

    A section is defined by a header and contains all content until the next
    header of the same or higher level.
    """

    title: str
    level: int  # Header level (1-6, 0 for no header)
    content: str  # Content without the header line
    start_pos: int  # Character position in original document
    end_pos: int
    children: list["Section"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Represents a parsed markdown document.

    Contains the extracted structure, metadata, and content from a markdown file.
    """

    path: str
    title: str | None
    content: str  # Body content without frontmatter
    metadata: dict[str, Any]  # Frontmatter metadata
    sections: list[Section]
    raw_content: str  # Original file content


# =============================================================================
# Computed Types (from transform pipeline)
# =============================================================================


@dataclass
class TextChunk:
    """
    Represents a chunk of text ready for embedding.

    Created by the chunker from parsed documents, with position tracking
    for source attribution.
    """

    content: str
    index: int  # Chunk index within document
    start_char: int  # Start position in source document
    end_char: int
    heading_path: str | None = None  # Hierarchical heading path (e.g., "Chapter 1 > Section 2")
    metadata: dict[str, Any] | None = None


# =============================================================================
# Storage Types (database records)
# =============================================================================


@dataclass
class Document:
    """
    Represents a document record in the database.

    Stores the full document content and metadata for retrieval and display.
    """

    id: int | None
    path: str
    title: str | None
    content: str
    metadata: dict[str, Any]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    file_mtime: float | None = None  # File modification time from os.stat().st_mtime


@dataclass
class Chunk:
    """
    Represents a chunk record in the database.

    Links to a parent document and stores the chunk content with its embedding.
    """

    id: int | None
    document_id: int
    content: str
    heading_path: str | None
    chunk_index: int
    start_char: int
    end_char: int
    embedding: list[float] | None = None


# =============================================================================
# Retrieval Types (search output)
# =============================================================================


@dataclass
class SearchResult:
    """
    Represents a search result returned by the retrieval system.

    Contains the matched chunk, its scores, and document context.
    """

    chunk_id: int
    document_id: int
    document_path: str
    content: str
    heading_path: str | None
    score: float  # Combined/final score
    vector_score: float | None = None  # Score from vector similarity
    fts_score: float | None = None  # Score from full-text search
    snippet: str | None = None  # Highlighted snippet for display

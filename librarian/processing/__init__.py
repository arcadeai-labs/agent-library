"""
Processing module for librarian.

Provides document parsing, chunking, and embedding functionality.

Usage:
    from librarian.processing import ProcessingManager
    from librarian.processing.embed import get_embedder
    from librarian.processing.parsers import MarkdownParser, ObsidianParser
"""

from librarian.processing.manager import ProcessingManager
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy

__all__ = [
    "Chunker",
    "ChunkingStrategy",
    "ProcessingManager",
]

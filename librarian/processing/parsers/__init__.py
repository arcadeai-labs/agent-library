"""
Document parsers for librarian.

Provides parsers for different document formats.

Usage:
    from librarian.processing.parsers import MarkdownParser, ObsidianParser
"""

from librarian.processing.parsers.base import (
    BaseParser,
    FileReadError,
    FileReadTimeoutError,
    safe_read_bytes,
    safe_read_text,
)
from librarian.processing.parsers.md import MarkdownParser
from librarian.processing.parsers.obsidian import ObsidianParser

__all__ = [
    "BaseParser",
    "FileReadError",
    "FileReadTimeoutError",
    "MarkdownParser",
    "ObsidianParser",
    "safe_read_bytes",
    "safe_read_text",
]

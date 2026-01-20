"""
Base parser interface.

All document parsers must implement this interface, enabling support
for different document formats (Markdown, Obsidian, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path

from librarian.types import ParsedDocument


class BaseParser(ABC):
    """
    Abstract base class for document parsers.

    Implementations handle parsing of specific document formats
    into a unified ParsedDocument representation.
    """

    @abstractmethod
    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document file.

        Args:
            file_path: Path to the document file.

        Returns:
            Parsed document with extracted structure and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file cannot be parsed.
        """
        ...

    @abstractmethod
    def parse_content(self, content: str, path: str = "") -> ParsedDocument:
        """
        Parse document content from a string.

        Args:
            content: The document content string.
            path: Optional path for reference in the ParsedDocument.

        Returns:
            Parsed document with extracted structure and metadata.
        """
        ...

    def can_parse(self, file_path: str | Path) -> bool:
        """
        Check if this parser can handle the given file.

        Default implementation checks file extension. Override for
        more sophisticated detection.

        Args:
            file_path: Path to check.

        Returns:
            True if this parser can handle the file.
        """
        return Path(file_path).suffix.lower() in self.supported_extensions

    @property
    def supported_extensions(self) -> set[str]:
        """
        Return the set of file extensions this parser supports.

        Override in subclasses to specify supported formats.

        Returns:
            Set of extension strings (e.g., {".md", ".markdown"}).
        """
        return set()

"""
Base parser interface.

All document parsers must implement this interface, enabling support
for different document formats (Markdown, Obsidian, etc.).
"""

import logging
import signal
from abc import ABC, abstractmethod
from pathlib import Path

from librarian.types import ParsedDocument

logger = logging.getLogger(__name__)

# Default timeout for file reads (seconds). Handles network/cloud filesystems
# (iCloud, Dropbox) where files may not be locally available.
DEFAULT_READ_TIMEOUT = 10


class FileReadError(OSError):
    """Raised when a file cannot be read due to I/O errors."""


class FileReadTimeoutError(FileReadError, TimeoutError):
    """Raised when a file read times out (e.g., iCloud file not synced)."""


def _timeout_handler(signum: int, frame: object) -> None:
    raise FileReadTimeoutError("File read timed out")


def safe_read_text(
    file_path: Path,
    timeout: int = DEFAULT_READ_TIMEOUT,
    encoding: str = "utf-8",
    fallback_encoding: str | None = "latin-1",
) -> str:
    """
    Read a text file with timeout protection and encoding fallback.

    Handles common issues with cloud-synced filesystems (iCloud, Dropbox)
    where files may not be locally available, causing read_text() to hang.

    Args:
        file_path: Path to the file.
        timeout: Max seconds to wait for the read.
        encoding: Primary encoding to try.
        fallback_encoding: Fallback encoding if primary fails. None to skip.

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        FileReadTimeoutError: If read exceeds timeout.
        FileReadError: For other I/O errors (permissions, etc.).
    """
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    old_handler = signal.getsignal(signal.SIGALRM)
    content: str | None = None
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

        try:
            content = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            if fallback_encoding:
                content = file_path.read_text(encoding=fallback_encoding)
            else:
                raise

        signal.alarm(0)
    except FileReadTimeoutError as e:
        raise FileReadTimeoutError(
            f"Timed out reading {file_path} after {timeout}s "
            f"(file may not be synced from cloud storage)"
        ) from e
    except FileNotFoundError:
        raise
    except PermissionError as e:
        raise FileReadError(f"Permission denied: {file_path}") from e
    except OSError as e:
        raise FileReadError(f"Cannot read {file_path}: {e}") from e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return content


def safe_read_bytes(
    file_path: Path,
    timeout: int = DEFAULT_READ_TIMEOUT,
) -> bytes:
    """
    Read a binary file with timeout protection.

    Args:
        file_path: Path to the file.
        timeout: Max seconds to wait for the read.

    Returns:
        File content as bytes.

    Raises:
        FileNotFoundError: If file doesn't exist.
        FileReadTimeoutError: If read exceeds timeout.
        FileReadError: For other I/O errors.
    """
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    old_handler = signal.getsignal(signal.SIGALRM)
    content: bytes | None = None
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        content = file_path.read_bytes()
        signal.alarm(0)
    except FileReadTimeoutError as e:
        raise FileReadTimeoutError(
            f"Timed out reading {file_path} after {timeout}s "
            f"(file may not be synced from cloud storage)"
        ) from e
    except FileNotFoundError:
        raise
    except PermissionError as e:
        raise FileReadError(f"Permission denied: {file_path}") from e
    except OSError as e:
        raise FileReadError(f"Cannot read {file_path}: {e}") from e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return content


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
            FileReadTimeoutError: If file read times out.
            FileReadError: For I/O errors.
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

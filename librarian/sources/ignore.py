"""
File ignore patterns for Librarian.

Handles .librarianignore files with gitignore-style pattern matching.
"""

import re
from pathlib import Path
from typing import ClassVar


class IgnorePatterns:
    """
    Manages file ignore patterns from .librarianignore files.

    Supports gitignore-style patterns:
    - Simple globs: *.py, *.md
    - Directory patterns: node_modules/, __pycache__/
    - Negation: !important.txt
    - Comments: # This is a comment
    """

    DEFAULT_PATTERNS: ClassVar[list[str]] = [
        # Version control
        ".git/",
        ".svn/",
        ".hg/",
        # Dependencies
        "node_modules/",
        "venv/",
        ".venv/",
        "env/",
        "__pycache__/",
        ".tox/",
        # Build outputs
        "build/",
        "dist/",
        "*.egg-info/",
        ".next/",
        ".nuxt/",
        "target/",
        # IDE
        ".vscode/",
        ".idea/",
        ".DS_Store",
        # Temporary files
        "*.tmp",
        "*.log",
        "*.cache",
        ".pytest_cache/",
        # Binary files
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.exe",
        # Large files
        "*.zip",
        "*.tar.gz",
        "*.tgz",
        "*.rar",
        # Media (usually too large for indexing)
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mp3",
        "*.wav",
    ]

    def __init__(self, root_path: Path, use_defaults: bool = True) -> None:
        """
        Initialize ignore patterns.

        Args:
            root_path: Root directory to search for .librarianignore files.
            use_defaults: Whether to include default ignore patterns.
        """
        self.root_path = root_path
        self.patterns: list[tuple[str, bool]] = []  # (pattern, is_negation)

        if use_defaults:
            for pattern in self.DEFAULT_PATTERNS:
                self.patterns.append((pattern, False))

        # Load .librarianignore if it exists
        ignore_file = root_path / ".librarianignore"
        if ignore_file.exists():
            self._load_ignore_file(ignore_file)

    def _load_ignore_file(self, ignore_file: Path) -> None:
        """
        Load patterns from a .librarianignore file.

        Args:
            ignore_file: Path to the ignore file.
        """
        content = ignore_file.read_text(encoding="utf-8")
        for line in content.split("\n"):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Check for negation
            is_negation = line.startswith("!")
            if is_negation:
                line = line[1:].strip()

            self.patterns.append((line, is_negation))

    def should_ignore(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored.

        Args:
            file_path: Path to check (absolute or relative to root).

        Returns:
            True if the file should be ignored.
        """
        # Get path relative to root
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            # Not relative to root, use as-is
            rel_path = file_path

        rel_path_str = str(rel_path)

        # Check patterns in order
        should_ignore = False
        for pattern, is_negation in self.patterns:
            if self._matches_pattern(rel_path_str, pattern):
                should_ignore = not is_negation

        return should_ignore

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches an ignore pattern.

        Args:
            path: File path to check.
            pattern: Ignore pattern.

        Returns:
            True if the path matches the pattern.
        """
        # Directory pattern (ends with /)
        if pattern.endswith("/"):
            # Match directory and all its contents
            pattern = pattern.rstrip("/")
            return path.startswith(f"{pattern}/") or path == pattern

        # Simple wildcard patterns
        if "*" in pattern:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace(".", "\\.").replace("*", ".*")
            regex_pattern = f"^{regex_pattern}$"
            return bool(re.match(regex_pattern, path))

        # Exact match
        return path == pattern


def should_ignore_file(file_path: Path, root_path: Path) -> bool:
    """
    Check if a file should be ignored based on patterns.

    Convenience function that creates an IgnorePatterns instance and checks.

    Args:
        file_path: Path to check.
        root_path: Root directory containing .librarianignore.

    Returns:
        True if the file should be ignored.
    """
    patterns = IgnorePatterns(root_path)
    return patterns.should_ignore(file_path)

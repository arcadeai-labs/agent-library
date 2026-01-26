"""
Source management for Librarian.

Manages different types of content sources: documents, codebases,
knowledge bases, and assets.
"""

from librarian.sources.codebase import CodebaseManager, detect_language, detect_primary_language
from librarian.sources.ignore import IgnorePatterns, should_ignore_file

__all__ = [
    "CodebaseManager",
    "IgnorePatterns",
    "detect_language",
    "detect_primary_language",
    "should_ignore_file",
]

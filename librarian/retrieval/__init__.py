"""
Retrieval module for librarian.

Provides hybrid search (vector + FTS) and MMR-based retrieval.
"""

from librarian.retrieval.search import HybridSearcher
from librarian.types import SearchResult

__all__ = [
    "HybridSearcher",
    "SearchResult",
]

"""
Hybrid search and MMR retrieval.

This module provides hybrid search combining vector similarity
and full-text search, with Max Marginal Relevance (MMR) for
diverse result selection.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from librarian.config import HYBRID_ALPHA, MMR_LAMBDA, SEARCH_LIMIT
from librarian.storage.database import get_database
from librarian.storage.fts_store import FTSStore
from librarian.storage.vector_store import VectorStore
from librarian.types import AssetType, SearchResult

if TYPE_CHECKING:
    from librarian.processing.embed import Embedder

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid searcher combining vector and full-text search.

    Supports:
    - Pure vector search
    - Pure full-text search
    - Hybrid search with configurable weighting
    - Max Marginal Relevance (MMR) for diverse results
    """

    def __init__(
        self,
        embedder: "Embedder",
        hybrid_alpha: float | None = None,
        mmr_lambda: float | None = None,
    ) -> None:
        """
        Initialize the hybrid searcher.

        Args:
            embedder: Embedder instance for generating query embeddings.
            hybrid_alpha: Weight for vector vs FTS (0=FTS only, 1=vector only).
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance).
        """
        self.embedder = embedder
        self.hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else HYBRID_ALPHA
        self.mmr_lambda = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA

        self.db = get_database()
        self.vector_store = VectorStore(self.db)
        self.fts_store = FTSStore(self.db)

    def search(
        self,
        query: str,
        limit: int | None = None,
        use_mmr: bool = True,
        filter_document_ids: list[int] | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            use_mmr: Whether to use MMR for diverse results.
            filter_document_ids: Optional list of document IDs to search within.

        Returns:
            List of search results ordered by relevance.
        """
        limit = limit or SEARCH_LIMIT

        # Get results from both search methods
        vector_results = self._vector_search(query, limit * 2)
        fts_results = self._fts_search(query, limit * 2)

        # Combine and score results
        combined = self._combine_results(vector_results, fts_results)

        # Filter by document IDs if specified
        if filter_document_ids:
            combined = [r for r in combined if r.document_id in filter_document_ids]

        # Apply MMR if requested
        if use_mmr and len(combined) > limit:
            combined = self._apply_mmr(query, combined, limit)
        else:
            combined = combined[:limit]

        return combined

    def vector_search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """
        Perform pure vector similarity search.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of search results.
        """
        limit = limit or SEARCH_LIMIT
        return self._vector_search(query, limit)

    def keyword_search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """
        Perform pure keyword/FTS search.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of search results.
        """
        limit = limit or SEARCH_LIMIT
        return self._fts_search(query, limit)

    def _vector_search(self, query: str, limit: int) -> list[SearchResult]:
        """Perform vector similarity search."""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, limit=limit)

        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=1.0 - r.distance,  # Convert distance to similarity
                vector_score=1.0 - r.distance,
                asset_type=AssetType(r.asset_type),
            )
            for r in results
        ]

    def _fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """Perform full-text search."""
        results = self.fts_store.search(query, limit=limit)

        # Normalize FTS scores (BM25 scores are negative, more negative = better)
        if not results:
            return []

        # Convert BM25 scores to positive similarities
        max_rank = max(abs(r.rank) for r in results) if results else 1.0
        if max_rank == 0:
            max_rank = 1.0

        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=1.0 - (abs(r.rank) / max_rank),  # Normalize to 0-1
                fts_score=1.0 - (abs(r.rank) / max_rank),
                snippet=r.snippet,
                asset_type=AssetType(r.asset_type) if r.asset_type else AssetType.TEXT,
            )
            for r in results
        ]

    def _combine_results(
        self,
        vector_results: list[SearchResult],
        fts_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Combine vector and FTS results with weighted scoring.

        Args:
            vector_results: Results from vector search.
            fts_results: Results from FTS search.

        Returns:
            Combined and scored results.
        """
        # Build lookup by chunk_id
        combined: dict[int, SearchResult] = {}

        # Add vector results
        for r in vector_results:
            combined[r.chunk_id] = SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=0.0,
                vector_score=r.vector_score,
                fts_score=None,
                asset_type=r.asset_type,
            )

        # Add/merge FTS results
        for r in fts_results:
            if r.chunk_id in combined:
                combined[r.chunk_id].fts_score = r.fts_score
                combined[r.chunk_id].snippet = r.snippet
            else:
                combined[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_path=r.document_path,
                    content=r.content,
                    heading_path=r.heading_path,
                    score=0.0,
                    vector_score=None,
                    fts_score=r.fts_score,
                    snippet=r.snippet,
                    asset_type=r.asset_type,
                )

        # Calculate combined scores
        for result in combined.values():
            vec_score = result.vector_score or 0.0
            fts_score = result.fts_score or 0.0

            # Weighted combination
            result.score = self.hybrid_alpha * vec_score + (1 - self.hybrid_alpha) * fts_score

        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: x.score, reverse=True)

        return results

    def _apply_mmr(
        self,
        query: str,
        candidates: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """
        Apply Max Marginal Relevance to select diverse results.

        MMR balances relevance to query with diversity among results.

        Args:
            query: The search query.
            candidates: Candidate results to select from.
            limit: Number of results to select.

        Returns:
            Diversified list of results.
        """
        if len(candidates) <= limit:
            return candidates

        # Get query embedding
        query_embedding = np.array(self.embedder.embed_query(query))

        # Get embeddings for all candidates
        candidate_embeddings: dict[int, np.ndarray] = {}
        for result in candidates:
            embedding = self.vector_store.get_embedding(result.chunk_id)
            if embedding:
                candidate_embeddings[result.chunk_id] = np.array(embedding)

        # If we don't have embeddings, fall back to simple ranking
        if not candidate_embeddings:
            return candidates[:limit]

        selected: list[SearchResult] = []
        remaining = list(candidates)

        while len(selected) < limit and remaining:
            best_score = -float("inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                if candidate.chunk_id not in candidate_embeddings:
                    continue

                cand_embedding = candidate_embeddings[candidate.chunk_id]

                # Relevance: similarity to query
                relevance = self._cosine_similarity(query_embedding, cand_embedding)

                # Diversity: max similarity to already selected
                if selected:
                    max_sim_to_selected = max(
                        self._cosine_similarity(
                            cand_embedding,
                            candidate_embeddings.get(s.chunk_id, np.zeros_like(cand_embedding)),
                        )
                        for s in selected
                        if s.chunk_id in candidate_embeddings
                    )
                else:
                    max_sim_to_selected = 0.0

                # MMR score
                mmr_score = (
                    self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        # Sort final selection by original score for display
        selected.sort(key=lambda x: x.score, reverse=True)

        return selected

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def search_in_document(
        self,
        query: str,
        document_id: int,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """
        Search within a specific document.

        Args:
            query: The search query.
            document_id: The document ID to search within.
            limit: Maximum number of results.

        Returns:
            List of search results from the document.
        """
        return self.search(
            query,
            limit=limit,
            use_mmr=False,
            filter_document_ids=[document_id],
        )

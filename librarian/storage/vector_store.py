"""
Vector store operations using sqlite-vec.

This module provides vector similarity search functionality
using the sqlite-vec extension.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from librarian.storage.database import get_effective_embedding_dimension, serialize_embedding

if TYPE_CHECKING:
    from librarian.storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""

    chunk_id: int
    distance: float
    content: str
    document_id: int
    document_path: str
    heading_path: str | None
    asset_type: str


class VectorStore:
    """
    Vector store operations using sqlite-vec.

    Uses cosine distance for normalized embeddings.
    Distance returned is 1 - cosine_similarity, so lower = more similar.
    """

    def __init__(self, database: "Database") -> None:
        """
        Initialize the vector store.

        Args:
            database: The database instance to use.
        """
        self.db = database

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: The query embedding vector.
            limit: Maximum number of results to return.
            min_similarity: Minimum similarity score (0-1) to include.

        Returns:
            List of search results ordered by similarity (most similar first).
        """
        expected_dim = get_effective_embedding_dimension()
        if len(query_embedding) != expected_dim:
            msg = (
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match expected {expected_dim}"
            )
            raise ValueError(msg)

        query_blob = serialize_embedding(query_embedding)

        with self.db.connection() as conn:
            # sqlite-vec uses distance (lower is more similar)
            # We convert to similarity for easier understanding
            rows = conn.execute(
                """
                SELECT
                    ce.chunk_id,
                    ce.distance,
                    c.content,
                    c.document_id,
                    c.heading_path,
                    d.path as document_path,
                    d.asset_type
                FROM chunk_embeddings ce
                JOIN chunks c ON ce.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE ce.embedding MATCH ?
                    AND k = ?
                ORDER BY ce.distance ASC
                """,
                (query_blob, limit * 2),  # Get extra for filtering
            ).fetchall()

            results = []
            for row in rows:
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1.0 - row["distance"]
                if similarity >= min_similarity:
                    results.append(
                        VectorSearchResult(
                            chunk_id=row["chunk_id"],
                            distance=row["distance"],
                            content=row["content"],
                            document_id=row["document_id"],
                            document_path=row["document_path"],
                            heading_path=row["heading_path"],
                            asset_type=row["asset_type"],
                        )
                    )
                    if len(results) >= limit:
                        break

            return results

    def search_with_exclusions(
        self,
        query_embedding: list[float],
        exclude_chunk_ids: set[int],
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """
        Search for similar chunks, excluding specified chunk IDs.

        This is useful for MMR (Maximal Marginal Relevance) where we need
        to find similar chunks while excluding already selected ones.

        Args:
            query_embedding: The query embedding vector.
            exclude_chunk_ids: Set of chunk IDs to exclude from results.
            limit: Maximum number of results to return.

        Returns:
            List of search results.
        """
        # Get more results than needed to account for exclusions
        results = self.search(query_embedding, limit=limit + len(exclude_chunk_ids))
        return [r for r in results if r.chunk_id not in exclude_chunk_ids][:limit]

    def get_embedding(self, chunk_id: int) -> list[float] | None:
        """
        Get the embedding for a specific chunk.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The embedding vector if found, None otherwise.
        """
        from librarian.storage.database import deserialize_embedding

        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row and row["embedding"]:
                return deserialize_embedding(row["embedding"])
            return None

    def update_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        """
        Update the embedding for a chunk.

        Args:
            chunk_id: The chunk ID.
            embedding: The new embedding vector.
        """
        with self.db._lock, self.db.connection() as conn:
            # Delete existing embedding if any
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
            # Insert new embedding
            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_embedding(embedding)),
            )

    def batch_update_embeddings(self, chunk_embeddings: list[tuple[int, list[float]]]) -> None:
        """
        Update embeddings for multiple chunks in a batch.

        Args:
            chunk_embeddings: List of (chunk_id, embedding) tuples.
        """
        with self.db._lock, self.db.connection() as conn:
            for chunk_id, embedding in chunk_embeddings:
                conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_embedding(embedding)),
                )

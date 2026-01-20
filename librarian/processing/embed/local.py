"""
Local embedding provider using sentence-transformers.

Provides embedding generation using locally-run sentence transformer models.
Supports lazy loading to avoid startup overhead when embeddings aren't needed.
"""

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from librarian.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from librarian.processing.embed.base import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using local sentence-transformers models.

    Lazily loads the model on first use to avoid startup costs.
    Thread-safe for concurrent embedding requests.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the local embedding provider.

        Args:
            model_name: Sentence transformer model name (default: from config).
            device: Device to run on ('cpu', 'cuda', 'mps', etc.).
        """
        self._model_name = model_name or EMBEDDING_MODEL
        self._device = device
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def model(self) -> "SentenceTransformer":
        """
        Lazily load and return the sentence transformer model.

        Thread-safe double-checked locking pattern.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        return self._model  # type: ignore[return-value]

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model: %s", self._model_name)

            if self._device:
                self._model = SentenceTransformer(self._model_name, device=self._device)
            else:
                self._model = SentenceTransformer(self._model_name)

            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Loaded model %s with dimension %d",
                self._model_name,
                self._dimension,
            )

        except ImportError as e:
            msg = (
                "sentence-transformers is required for local embedding. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(msg) from e

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading model if needed."""
        if self._dimension is None:
            _ = self.model  # Trigger load
        return self._dimension or EMBEDDING_DIMENSION

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        result: list[float] = embedding.tolist()
        return result

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for processing.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """
        Generate query-optimized embedding.

        Handles model-specific query prefixes (E5, Instructor models).
        """
        # E5 models expect "query:" prefix
        if "e5" in self._model_name.lower():
            query = f"query: {query}"
        # Instructor models use instruction prefix
        elif "instructor" in self._model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        return self.embed(query)

    def embed_document(self, document: str) -> list[float]:
        """
        Generate document-optimized embedding.

        Handles model-specific document prefixes.
        """
        if "e5" in self._model_name.lower():
            document = f"passage: {document}"

        return self.embed(document)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate document-optimized embeddings for multiple documents.

        Handles model-specific document prefixes.
        """
        # Apply prefixes for models that need them
        if "e5" in self._model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]

        return self.embed_batch(documents)

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (0-1).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

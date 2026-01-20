"""
Base embedding provider interface.

All embedding providers must implement this interface, enabling swappable
backends (local sentence-transformers, OpenAI, etc.) without changing
the rest of the codebase.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implementations handle the actual embedding generation using different
    backends (local models, API services, etc.).
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the embedding dimension for this provider.

        Returns:
            Number of dimensions in the embedding vectors.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the model identifier used by this provider.

        Returns:
            Model name/identifier string.
        """
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Implementations should optimize for batch processing where possible.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding optimized for search queries.

        Some models use different encodings for queries vs documents.
        Default implementation calls embed().

        Args:
            query: Query text to embed.

        Returns:
            Query embedding vector.
        """
        return self.embed(query)

    def embed_document(self, document: str) -> list[float]:
        """
        Generate embedding optimized for documents/passages.

        Some models use different encodings for queries vs documents.
        Default implementation calls embed().

        Args:
            document: Document text to embed.

        Returns:
            Document embedding vector.
        """
        return self.embed(document)

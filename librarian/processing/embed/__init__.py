"""
Embedding module for librarian.

Provides a unified interface for embedding generation with swappable backends.
Supports local sentence-transformers and OpenAI API.

Usage:
    from librarian.processing.embed import get_embedder, Embedder

    # Get default embedder (based on config)
    embedder = get_embedder()

    # Embed text
    vector = embedder.embed("Hello world")
    vectors = embedder.embed_batch(["Hello", "World"])
"""

import logging
import threading

from librarian.config import EMBEDDING_PROVIDER
from librarian.processing.embed.base import EmbeddingProvider
from librarian.processing.embed.local import LocalEmbeddingProvider
from librarian.processing.embed.openai import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)

__all__ = [
    "Embedder",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedder",
]


class Embedder:
    """
    Unified embedding interface wrapping an EmbeddingProvider.

    Provides a consistent API regardless of the underlying provider.
    Delegates all embedding operations to the provider.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """
        Initialize with a provider.

        Args:
            provider: The embedding provider to use.
        """
        self._provider = provider

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the underlying provider."""
        return self._provider

    @property
    def model_name(self) -> str:
        """Return the model name from the provider."""
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._provider.dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return self._provider.embed_batch(texts)

    def embed_query(self, query: str) -> list[float]:
        """Generate query-optimized embedding."""
        return self._provider.embed_query(query)

    def embed_document(self, document: str) -> list[float]:
        """Generate document-optimized embedding."""
        return self._provider.embed_document(document)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate document-optimized embeddings for multiple documents."""
        if hasattr(self._provider, "embed_documents"):
            result: list[list[float]] = self._provider.embed_documents(documents)
            return result
        # Fallback to embed_batch if provider doesn't have embed_documents
        return self._provider.embed_batch(documents)


def _create_provider(provider_type: str | None = None) -> EmbeddingProvider:
    """
    Create an embedding provider based on type.

    Args:
        provider_type: Provider type ("local" or "openai"). Defaults to config.

    Returns:
        Configured EmbeddingProvider instance.

    Raises:
        ValueError: If provider type is unknown.
    """
    provider_type = provider_type or EMBEDDING_PROVIDER

    if provider_type == "local":
        logger.info("Using local embedding provider (sentence-transformers)")
        return LocalEmbeddingProvider()

    if provider_type == "openai":
        logger.info("Using OpenAI embedding provider")
        return OpenAIEmbeddingProvider()

    msg = f"Unknown embedding provider: {provider_type}. Use 'local' or 'openai'."
    raise ValueError(msg)


# Global embedder instance (singleton pattern)
_embedder_instance: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(provider_type: str | None = None) -> Embedder:
    """
    Get the global Embedder instance.

    Creates a new embedder if one doesn't exist or if a different
    provider type is requested.

    Args:
        provider_type: Optional provider type to use. If different from
                      current provider, a new embedder is created.

    Returns:
        The global Embedder instance.
    """
    global _embedder_instance

    requested_type = provider_type or EMBEDDING_PROVIDER

    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                provider = _create_provider(requested_type)
                _embedder_instance = Embedder(provider)

    # Check if we need a different provider type
    elif provider_type:
        current_is_local = isinstance(_embedder_instance.provider, LocalEmbeddingProvider)
        requested_is_local = requested_type == "local"

        if current_is_local != requested_is_local:
            with _embedder_lock:
                provider = _create_provider(requested_type)
                _embedder_instance = Embedder(provider)

    return _embedder_instance


def reset_embedder() -> None:
    """
    Reset the global embedder instance.

    Useful for testing or when switching providers.
    """
    global _embedder_instance
    with _embedder_lock:
        _embedder_instance = None

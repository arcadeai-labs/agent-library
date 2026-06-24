"""
Embedding module for librarian.

Provides a unified interface for embedding generation with swappable backends.
Supports local sentence-transformers and OpenAI API, with multi-modal support
for text, code (CodeBERT), and vision (CLIP) embeddings.

Usage:
    from librarian.processing.embed import get_embedder, Embedder

    # Get default embedder (based on config)
    embedder = get_embedder()

    # Embed text
    vector = embedder.embed("Hello world")
    vectors = embedder.embed_batch(["Hello", "World"])

    # Get modality-specific embedder
    from librarian.types import EmbeddingModality
    code_embedder = get_embedder_for_modality(EmbeddingModality.CODE)
    if code_embedder:
        code_vector = code_embedder.embed("def hello(): pass")
"""

import logging
import threading
from typing import TYPE_CHECKING

from librarian.config import (
    CODE_EMBEDDING_MODEL,
    CODE_EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDER,
    ENABLE_CODE_EMBEDDINGS,
    ENABLE_VISION_EMBEDDINGS,
    VISION_EMBEDDING_MODEL,
)
from librarian.processing.embed.base import EmbeddingProvider
from librarian.processing.embed.openai import OpenAIEmbeddingProvider
from librarian.types import EmbeddingModality

if TYPE_CHECKING:
    from librarian.processing.embed.local import LocalEmbeddingProvider, ModelLoadTimeoutError

logger = logging.getLogger(__name__)

__all__ = [
    "Embedder",
    "EmbeddingProvider",
    "EmbeddingRegistry",
    "LocalEmbeddingProvider",
    "ModelLoadTimeoutError",
    "OpenAIEmbeddingProvider",
    "get_embedder",
    "get_embedder_for_modality",
    "get_embedding_registry",
]


def __getattr__(name: str) -> object:
    """Lazily expose local embedding classes without importing ML dependencies."""
    if name in {"LocalEmbeddingProvider", "ModelLoadTimeoutError"}:
        from librarian.processing.embed.local import LocalEmbeddingProvider, ModelLoadTimeoutError

        lazy_exports: dict[str, object] = {
            "LocalEmbeddingProvider": LocalEmbeddingProvider,
            "ModelLoadTimeoutError": ModelLoadTimeoutError,
        }
        return lazy_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


class EmbeddingRegistry:
    """
    Registry managing embedders per modality.

    Provides lazy loading of modality-specific embedders (text, code, vision)
    based on configuration flags. Ensures each modality has at most one
    embedder instance.

    Usage:
        registry = get_embedding_registry()
        code_embedder = registry.get_embedder(EmbeddingModality.CODE)
        if code_embedder:
            vector = code_embedder.embed("def hello(): pass")
    """

    def __init__(self) -> None:
        """Initialize the embedding registry."""
        self._embedders: dict[EmbeddingModality, Embedder] = {}
        self._unavailable: set[EmbeddingModality] = set()
        self._lock = threading.Lock()

    def is_enabled(self, modality: EmbeddingModality) -> bool:
        """
        Check if a modality is enabled in configuration.

        Args:
            modality: The embedding modality to check.

        Returns:
            True if the modality is enabled, False otherwise.
        """
        if modality == EmbeddingModality.TEXT:
            return True  # Text is always enabled
        elif modality == EmbeddingModality.CODE:
            return ENABLE_CODE_EMBEDDINGS
        elif modality == EmbeddingModality.VISION:
            return ENABLE_VISION_EMBEDDINGS
        return False

    def get_embedder(self, modality: EmbeddingModality) -> Embedder | None:
        """
        Get embedder for a specific modality, creating if needed.

        Args:
            modality: The embedding modality to get an embedder for.

        Returns:
            Embedder instance if modality is enabled, None otherwise.
        """
        if not self.is_enabled(modality):
            logger.debug(f"Modality {modality.value} is not enabled")
            return None

        if modality in self._embedders:
            return self._embedders[modality]

        if modality in self._unavailable:
            return None

        with self._lock:
            # Double-check after acquiring lock
            if modality in self._embedders:
                return self._embedders[modality]
            if modality in self._unavailable:
                return None

            embedder = self._create_embedder_for_modality(modality)
            if embedder:
                self._embedders[modality] = embedder
            else:
                self._unavailable.add(modality)
            return embedder

    def _create_embedder_for_modality(self, modality: EmbeddingModality) -> Embedder | None:
        """
        Create an embedder for a specific modality.

        Eagerly validates that required dependencies are installed for CODE
        and VISION modalities. If validation fails, logs a warning and returns
        None (the modality will be silently skipped during search).

        Args:
            modality: The embedding modality.

        Returns:
            Configured Embedder instance, or None if dependencies are missing.
        """
        if modality == EmbeddingModality.TEXT:
            provider = _create_provider()
            logger.info(f"Created TEXT embedder with model: {provider.model_name}")
            return Embedder(provider)

        elif modality == EmbeddingModality.CODE:
            if not ENABLE_CODE_EMBEDDINGS:
                return None
            try:
                from librarian.processing.embed.local import LocalEmbeddingProvider

                if CODE_EMBEDDING_PROVIDER == "local":
                    provider = LocalEmbeddingProvider(model_name=CODE_EMBEDDING_MODEL)
                else:
                    provider = LocalEmbeddingProvider(model_name=CODE_EMBEDDING_MODEL)
                # Eagerly validate dependencies (does NOT load the model)
                provider.validate()
                logger.info(f"Created CODE embedder with model: {CODE_EMBEDDING_MODEL}")
                return Embedder(provider)
            except ImportError as e:
                logger.debug(
                    "CODE embeddings not available (optional dependency missing: %s). "
                    "Code content will use TEXT embeddings instead.",
                    e,
                )
                return None

        elif modality == EmbeddingModality.VISION:
            if not ENABLE_VISION_EMBEDDINGS:
                return None
            try:
                from librarian.processing.embed.local import LocalEmbeddingProvider

                provider = LocalEmbeddingProvider(model_name=VISION_EMBEDDING_MODEL)
                # Eagerly validate dependencies (does NOT load the model)
                provider.validate()
                logger.info(f"Created VISION embedder with model: {VISION_EMBEDDING_MODEL}")
                return Embedder(provider)
            except ImportError as e:
                logger.debug(
                    "VISION embeddings not available (optional dependency missing: %s). "
                    "Image content will use TEXT embeddings instead.",
                    e,
                )
                return None

        return None

    def ensure_models(self) -> dict[str, str]:
        """
        Ensure all enabled embedding models are downloaded and loadable.

        Triggers actual model loading for each enabled modality.
        Models are downloaded from HuggingFace Hub if not already cached.

        Returns:
            Dict mapping modality name to status string.
            Example: {"text": "loaded (384d)", "code": "loaded (768d)",
                      "vision": "unavailable (missing dependencies)"}
        """
        results: dict[str, str] = {}

        for modality in [EmbeddingModality.TEXT, EmbeddingModality.CODE, EmbeddingModality.VISION]:
            if not self.is_enabled(modality):
                results[modality.value] = "disabled"
                continue

            try:
                embedder = self.get_embedder(modality)
                if embedder is None:
                    results[modality.value] = "unavailable (missing dependencies)"
                    continue
                # Trigger actual model loading by accessing dimension
                dim = embedder.dimension
                results[modality.value] = f"loaded ({dim}d)"
            except ImportError as e:
                results[modality.value] = f"missing dependencies: {e}"
            except Exception as e:
                results[modality.value] = f"error: {e}"

        return results

    def reset(self) -> None:
        """Reset all embedders. Useful for testing."""
        with self._lock:
            self._embedders.clear()
            self._unavailable.clear()


# Global registry instance (singleton)
_registry_instance: EmbeddingRegistry | None = None
_registry_lock = threading.Lock()


def get_embedding_registry() -> EmbeddingRegistry:
    """
    Get the global EmbeddingRegistry instance.

    Returns:
        The global EmbeddingRegistry singleton.
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = EmbeddingRegistry()

    return _registry_instance


def get_embedder_for_modality(modality: EmbeddingModality) -> Embedder | None:
    """
    Convenience function to get embedder for a specific modality.

    Args:
        modality: The embedding modality (TEXT, CODE, VISION).

    Returns:
        Embedder instance if modality is enabled, None otherwise.

    Example:
        code_embedder = get_embedder_for_modality(EmbeddingModality.CODE)
        if code_embedder:
            embedding = code_embedder.embed("def hello(): pass")
    """
    registry = get_embedding_registry()
    return registry.get_embedder(modality)


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
        from librarian.processing.embed.local import LocalEmbeddingProvider

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
        current_is_local = _embedder_instance.provider.__class__.__name__ == "LocalEmbeddingProvider"
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

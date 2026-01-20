"""
OpenAI-compatible embedding provider.

Provides embedding generation using any OpenAI-compatible API with efficient
batch processing. Default: local qwen3-embedding service at localhost:7171.

Supports both standard OpenAI API (using 'input') and custom servers (using 'texts').
Supports instruction-based embeddings for models like Qwen3-Embedding.
"""

import logging
from dataclasses import dataclass

import httpx

from librarian.config import (
    EMBEDDING_QUERY_INSTRUCTION,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_BATCH_SIZE,
    OPENAI_EMBEDDING_DIMENSION,
    OPENAI_EMBEDDING_MODEL,
)
from librarian.processing.embed.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingData:
    """Single embedding result."""

    index: int
    embedding: list[float]


@dataclass
class EmbeddingResponse:
    """Response from embedding API."""

    data: list[EmbeddingData]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI-compatible API.

    Works with OpenAI, local llama.cpp, vLLM, GoEmbed, or any compatible server.
    Uses raw HTTP to support servers that expect 'texts' instead of 'input'.
    Supports instruction-based embeddings for models like Qwen3-Embedding.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
        query_instruction: str | None = None,
    ) -> None:
        """
        Initialize the OpenAI-compatible embedding provider.

        Args:
            api_key: API key (default: from config, "not-needed" for local).
            base_url: API base URL (default: http://mansamura:7171/v1).
            model: Embedding model name (default: qwen3-embedding-06b).
            dimension: Embedding dimension (default: 1024).
            batch_size: Max texts per API call (default: 64).
            query_instruction: Task description for query embeddings.
        """
        self._api_key = api_key or OPENAI_API_KEY
        self._base_url = (base_url or OPENAI_API_BASE).rstrip("/")
        self._model_name = model or OPENAI_EMBEDDING_MODEL
        self._dimension = dimension or OPENAI_EMBEDDING_DIMENSION
        self._batch_size = batch_size or OPENAI_EMBEDDING_BATCH_SIZE
        self._query_instruction = query_instruction or EMBEDDING_QUERY_INSTRUCTION
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=120.0)
            logger.info("Initialized embedding client for %s", self._base_url)
        return self._client

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def _call_api(
        self,
        texts: list[str],
        instruction_type: str | None = None,
        task: str | None = None,
    ) -> EmbeddingResponse:
        """
        Call the embedding API.

        Args:
            texts: List of texts to embed.
            instruction_type: "query" or "document" for instruction-based models.
            task: Task description for query instructions.

        Returns:
            EmbeddingResponse with embeddings for each text.
        """
        url = f"{self._base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict = {"model": self._model_name, "texts": texts}

        # Add instruction config for instruction-based models (e.g., Qwen3-Embedding)
        # GoEmbed InstructionConfig: type, task, context, enabled, template_style
        if instruction_type:
            instruction_config: dict = {"type": instruction_type}
            if task:
                instruction_config["task"] = task
            payload["instruction"] = instruction_config

        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        # Parse response - handle both OpenAI format and simple list format
        if "data" in data:
            # OpenAI format: {"data": [{"index": 0, "embedding": [...]}]}
            embeddings = [
                EmbeddingData(index=item.get("index", i), embedding=item["embedding"])
                for i, item in enumerate(data["data"])
            ]
        elif "embeddings" in data:
            # Simple format: {"embeddings": [[...], [...]]}
            embeddings = [
                EmbeddingData(index=i, embedding=emb) for i, emb in enumerate(data["embeddings"])
            ]
        else:
            msg = f"Unexpected response format: {list(data.keys())}"
            raise ValueError(msg)

        return EmbeddingResponse(data=embeddings)

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (no instruction).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self._call_api([text])
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching (no instruction).

        Splits large requests into batches to respect API limits.
        Maintains input order in output.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order as input.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._call_api(batch)

            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Uses query instruction for instruction-based models (Qwen3-Embedding).

        Args:
            query: Search query text.

        Returns:
            Query embedding vector.
        """
        response = self._call_api(
            [query],
            instruction_type="query",
            task=self._query_instruction or None,
        )
        return response.data[0].embedding

    def embed_document(self, document: str) -> list[float]:
        """
        Generate embedding for a document.

        Uses document instruction type for instruction-based models.

        Args:
            document: Document text to embed.

        Returns:
            Document embedding vector.
        """
        response = self._call_api([document], instruction_type="document")
        return response.data[0].embedding

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Uses document instruction type for instruction-based models.

        Args:
            documents: List of document texts.

        Returns:
            List of document embedding vectors.
        """
        if not documents:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            response = self._call_api(batch, instruction_type="document")

            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

"""
Pytest configuration and shared fixtures.

Provides fixtures for testing with fake embeddings for fast tests.
"""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Set test environment variables BEFORE importing librarian modules
os.environ["DATABASE_PATH"] = "/tmp/librarian_test.db"  # noqa: S108
os.environ["DOCUMENTS_PATH"] = "/tmp/librarian_test_docs"  # noqa: S108
os.environ["EMBEDDING_PROVIDER"] = "local"  # Use local embedder in tests (384 dimensions)
os.environ["EMBEDDING_DIMENSION"] = "384"  # Match fake embedder dimension


class FakeEmbeddingProvider:
    """
    Fake embedding provider for tests.

    Uses text hashing to generate consistent embeddings for the same input.
    """

    def __init__(self, model_name: str = "fake-model", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_to_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self._dimension).astype(np.float32).tolist()

    def embed(self, text: str) -> list[float]:
        return self._hash_to_embedding(text)

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_embedding(f"query:{query}")

    def embed_document(self, document: str) -> list[float]:
        return self._hash_to_embedding(f"doc:{document}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_embedding(t) for t in texts]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate document-optimized embeddings."""
        return [self._hash_to_embedding(f"doc:{d}") for d in documents]

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))


class FakeEmbedder:
    """
    Fake embedder wrapping a fake provider for tests.

    Matches the real Embedder interface.
    """

    def __init__(self, provider: FakeEmbeddingProvider | None = None) -> None:
        self._provider = provider or FakeEmbeddingProvider()

    @property
    def provider(self) -> FakeEmbeddingProvider:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, text: str) -> list[float]:
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed_batch(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._provider.embed_query(query)

    def embed_document(self, document: str) -> list[float]:
        return self._provider.embed_document(document)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._provider.embed_documents(documents)


# Global fake embedder instance for tests
_fake_provider = FakeEmbeddingProvider()
_fake_embedder = FakeEmbedder(_fake_provider)


def get_fake_embedder(provider_type: str | None = None) -> FakeEmbedder:
    """Return the fake embedder for tests."""
    return _fake_embedder


@pytest.fixture(scope="session", autouse=True)
def patch_embedder_globally() -> Generator[None, None, None]:
    """
    Patch the embedder globally for all tests at session scope.

    This ensures no test ever loads the real sentence transformer model.
    """
    # Patch the new embed module location
    with (
        patch("librarian.processing.embed.get_embedder", get_fake_embedder),
        patch("librarian.processing.embed.Embedder", FakeEmbedder),
        patch("librarian.processing.embed._embedder_instance", _fake_embedder),
        patch("librarian.server.get_embedder", get_fake_embedder),
        patch("librarian.processing.embed.local.LocalEmbeddingProvider", FakeEmbeddingProvider),
    ):
        yield


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Get the fake embedder instance for tests that need direct access."""
    return _fake_embedder


@pytest.fixture(scope="session")
def sample_markdown_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create sample markdown files for testing."""
    base_tmp = tmp_path_factory.mktemp("sample_docs")

    (base_tmp / "simple.md").write_text(
        "# Simple Document\n\nThis is a simple markdown document.\n"
    )

    (base_tmp / "with_frontmatter.md").write_text(
        "---\ntitle: Document with Frontmatter\ntags:\n  - test\n---\n\n# Header\n\nContent.\n"
    )

    (base_tmp / "multi_section.md").write_text(
        "# Main Title\n\nIntro.\n\n## First Section\n\nFirst content.\n\n"
        "## Second Section\n\nSecond content.\n"
    )

    (base_tmp / "long_document.md").write_text(
        "# Long Document\n\n" + "This is paragraph content. " * 100 + "\n"
    )

    return base_tmp


@pytest.fixture
def temp_database_path(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_index.db")


@pytest.fixture
def clean_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Create a clean temporary database for each test."""
    from librarian import config as config_module
    from librarian.storage import database as db_module

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))

    # Reset the global database instance
    db_module._db_instance = None

    yield db_path

    # Cleanup
    db_module._db_instance = None
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_docs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary documents directory with test files."""
    from librarian import config as config_module

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    (docs_dir / "test1.md").write_text("# Test Document 1\n\nContent for test 1.")
    (docs_dir / "test2.md").write_text("---\ntitle: Test Document 2\n---\n\nContent for test 2.")
    (docs_dir / "subdir").mkdir()
    (docs_dir / "subdir" / "nested.md").write_text("# Nested Document\n\nNested content.")

    monkeypatch.setattr(config_module, "DOCUMENTS_PATH", str(docs_dir))

    return docs_dir

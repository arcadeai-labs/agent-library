"""
Dialect-free storage helpers shared by every concrete backend.

These have zero substrate content -- they are pure value transforms used
identically by :mod:`librarian.storage.sqlite_storage` and
:mod:`librarian.storage.postgres`. Keeping them here stops the byte-for-byte
twins from drifting. (The genuinely dialect-specific SQL -- ``write_upsert`` /
``transaction`` / ``put_sync_state`` -- intentionally stays duplicated per
backend, since its body diverges.)
"""

from datetime import date, datetime
from typing import Any

from librarian.types import EmbeddingModality

__all__ = ["iso", "json_default", "modality_table"]


def modality_table(modality: EmbeddingModality) -> str:
    """Map an embedding modality to its physical embedding table name."""
    if modality == EmbeddingModality.CODE:
        return "vec_chunks_code"
    if modality == EmbeddingModality.VISION:
        return "vec_chunks_vision"
    return "chunk_embeddings"


def iso(value: datetime | None) -> str | None:
    """Render a datetime as an ISO-8601 string, passing ``None`` through."""
    return value.isoformat() if value is not None else None


def json_default(value: Any) -> str:
    """JSON fallback for date/datetime values (e.g. from YAML frontmatter)."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)

"""
Dialect-free storage helpers shared by every concrete backend.

These have zero substrate content -- they are pure value transforms used
identically by :mod:`librarian.storage.sqlite_storage` and
:mod:`librarian.storage.postgres`. Keeping them here stops the byte-for-byte
twins from drifting. (The genuinely dialect-specific SQL -- ``write_upsert`` /
``transaction`` / ``put_sync_state`` -- intentionally stays duplicated per
backend, since its body diverges.)
"""

import re
from datetime import date, datetime
from typing import Any

from librarian.types import EmbeddingModality

__all__ = ["iso", "json_default", "modality_table", "validate_json_key"]

# A ``modality_data`` JSON key is interpolated directly into a JSON path
# expression in both backends' reprocess queries (sqlite ``json_extract`` and
# Postgres ``->>``), so restrict it to a bare identifier to keep it injection-safe.
_JSON_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_json_key(key: str) -> str:
    """Return ``key`` if it is a safe JSON object key, else raise ``ValueError``.

    Used to guard keys that are interpolated into a JSON-path SQL expression
    (they cannot be bound as parameters). Only plain identifiers are allowed.
    """
    if not _JSON_KEY_RE.match(key):
        raise ValueError(
            f"Invalid modality_data key {key!r}; expected a bare identifier "
            "such as 'processing_status'."
        )
    return key


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

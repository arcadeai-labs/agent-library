"""
Deterministic identity hashing for documents and chunks.

v0.14 makes document and chunk identity a deterministic function of the
source that produced them, rather than an auto-incrementing database row id.
This is what makes idempotent upsert possible without a "find then update"
round trip: re-ingesting edited content recomputes the same id and overwrites
in place.

Identity is ``hash(connector_name, source_type, source_native_id)``. The
source-native id is whatever uniquely identifies the document (or chunk)
within the connector's source -- an absolute file path, a Slack message ts,
a GitHub review-comment id, etc.
"""

import hashlib

__all__ = ["chunk_id", "document_id"]

# A byte that cannot appear in any UTF-8 field value, used to separate the
# components so that ("a", "bc") and ("ab", "c") never collide.
_SEP = "\x00"


def _stable_hash(connector_name: str, source_type: str, source_native_id: str) -> str:
    """Return a stable hex digest for the given identity tuple.

    Uses SHA-256 (not Python's built-in ``hash()``, which is salted per process
    and therefore not stable across runs).
    """
    payload = _SEP.join((connector_name, source_type, source_native_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def document_id(connector_name: str, source_type: str, source_native_id: str) -> str:
    """Deterministic document id.

    Args:
        connector_name: The owning connector's name (e.g. ``"local_file"``).
        source_type: The source type within the connector (e.g. ``"file"``).
        source_native_id: The id that uniquely identifies the document within
            the source (e.g. an absolute file path).

    Returns:
        A 64-char hex digest stable across processes and runs.
    """
    return _stable_hash(connector_name, source_type, source_native_id)


def chunk_id(connector_name: str, source_type: str, source_native_id: str) -> str:
    """Deterministic chunk id.

    Args:
        connector_name: The owning connector's name.
        source_type: The source type within the connector.
        source_native_id: The id that uniquely identifies the chunk within the
            source (e.g. ``"<file path>#chunk=3"``).

    Returns:
        A 64-char hex digest stable across processes and runs.
    """
    return _stable_hash(connector_name, source_type, source_native_id)

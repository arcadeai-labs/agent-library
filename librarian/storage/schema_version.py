"""
v0.13 -> v0.14 schema detection and the detect-and-rebuild guard.

v0.14 changed the on-disk schema in a way that cannot be migrated in place for an
already-populated v0.13 database (the ``chunk_embeddings`` vec0 table gains a
``model_version`` auxiliary column, and vec0 tables cannot be ``ALTER``-ed). Rather
than silently corrupt or half-migrate, v0.14 detects an old database on startup,
refuses to run normally, and directs the user to back up and run
``libr index --rebuild``.

This module is intentionally dependency-light: detection runs over a plain
``sqlite3`` connection (no sqlite-vec load required) so it is cheap to call on
every CLI command.
"""

import sqlite3
from pathlib import Path
from typing import Literal

SchemaVersion = Literal["empty", "v0.13", "v0.14"]


class SchemaRebuildRequired(RuntimeError):
    """Raised when a populated v0.13 database is found and must be rebuilt."""


REBUILD_MESSAGE = (
    "Your library database uses the pre-v0.14 schema and can't be upgraded in "
    "place.\n\n"
    "Back up the file if you want to keep it, then rebuild:\n"
    "    libr index --rebuild\n\n"
    "The rebuild auto-backs-up the old database to "
    "<db>.v0-backup (pass --no-backup to skip), wipes the index, recreates it "
    "under the v0.14 schema, and re-ingests your configured sources."
)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    return {row[0] for row in rows}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {row[1] for row in rows}


def _embeddings_has_model_version(conn: sqlite3.Connection) -> bool:
    # chunk_embeddings is a sqlite-vec ``vec0`` virtual table; it cannot be queried
    # without the extension loaded. Detection runs over a plain connection, so we
    # inspect the stored CREATE statement instead of querying the table.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_embeddings'"
    ).fetchone()
    return bool(row and row[0] and "model_version" in row[0])


def detect_schema_version(conn: sqlite3.Connection) -> SchemaVersion:
    """Classify the schema reachable through ``conn``.

    * ``"empty"``  -- no librarian data yet (no ``documents`` table, or it has no
      rows). Safe to migrate forward automatically.
    * ``"v0.14"`` -- carries all v0.14 markers (``chunks.chunk_id`` column, the
      ``sync_state`` table, and ``chunk_embeddings.model_version``).
    * ``"v0.13"`` -- has indexed data but lacks the v0.14 markers; requires a
      rebuild.
    """
    tables = _table_names(conn)
    if "documents" not in tables:
        return "empty"

    has_v14_markers = (
        "chunk_id" in _columns(conn, "chunks")
        and "sync_state" in tables
        and _embeddings_has_model_version(conn)
    )
    if has_v14_markers:
        return "v0.14"

    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if not doc_count:
        # Base tables exist but nothing has been ingested yet: a fresh database
        # that simply hasn't been migrated forward. Safe to treat as empty.
        return "empty"

    return "v0.13"


def schema_status_for_path(db_path: str | Path) -> SchemaVersion:
    """Detect the schema version of an on-disk database (``"empty"`` if absent)."""
    path = Path(db_path)
    if not path.exists():
        return "empty"
    conn = sqlite3.connect(str(path))
    try:
        return detect_schema_version(conn)
    finally:
        conn.close()

"""Tests for the v0.13 -> v0.14 schema detection and rebuild gating.

These cover the detect-and-rebuild boundary: a fresh/empty or already-v0.14
database migrates forward transparently, while a *populated* v0.13 database is
detected and refused (migrate raises ``SchemaRebuildRequired``) so the user is
steered to ``libr index --rebuild``.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from librarian.storage.database import get_database, serialize_embedding
from librarian.storage.migrate import migrate
from librarian.storage.schema_version import (
    SchemaRebuildRequired,
    detect_schema_version,
    schema_status_for_path,
)


def _insert_legacy_embedding(db) -> None:  # type: ignore[no-untyped-def]
    """Write one row into the base (v0.13) chunk_embeddings table.

    This makes the table non-empty *without* a ``model_version`` column, which is
    exactly the state migrate() cannot upgrade in place.
    """
    zero = serialize_embedding([0.0] * 384)
    with db._connection() as conn:
        conn.execute(
            "INSERT INTO documents (path, title, content) VALUES (?, ?, ?)",
            ("/legacy/doc.md", "Legacy", "legacy content"),
        )
        conn.execute(
            "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
            (1, zero),
        )


class TestSchemaDetection:
    def test_absent_database_is_empty(self, tmp_path: Path) -> None:
        assert schema_status_for_path(tmp_path / "nope.db") == "empty"

    def test_bare_database_is_empty(self, tmp_path: Path) -> None:
        db_file = tmp_path / "bare.db"
        sqlite3.connect(str(db_file)).close()
        assert schema_status_for_path(db_file) == "empty"

    def test_fresh_base_schema_is_empty(self, clean_db: Path) -> None:
        # Base tables exist but nothing has been ingested yet.
        get_database()
        with closing(sqlite3.connect(str(clean_db))) as conn:
            assert detect_schema_version(conn) == "empty"

    def test_migrated_database_is_v014(self, clean_db: Path) -> None:
        from librarian.storage.sqlite_storage import SQLiteStorage

        SQLiteStorage(database=get_database()).migrate()
        with closing(sqlite3.connect(str(clean_db))) as conn:
            assert detect_schema_version(conn) == "v0.14"

    def test_populated_legacy_database_is_v013(self, clean_db: Path) -> None:
        db = get_database()
        _insert_legacy_embedding(db)
        with closing(sqlite3.connect(str(clean_db))) as conn:
            assert detect_schema_version(conn) == "v0.13"


class TestMigrateGating:
    def test_migrate_raises_on_populated_legacy_embeddings(self, clean_db: Path) -> None:
        db = get_database()
        _insert_legacy_embedding(db)
        with db._connection() as conn, pytest.raises(SchemaRebuildRequired):
            migrate(conn)

    def test_migrate_succeeds_on_empty_database(self, clean_db: Path) -> None:
        db = get_database()
        with db._connection() as conn:
            migrate(conn)  # should not raise
            assert detect_schema_version(conn) == "v0.14"


class TestCliGuard:
    def test_guarded_command_refuses_on_v013(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A normal command aborts loudly when the DB is pre-v0.14."""
        from typer.testing import CliRunner

        from librarian import config as config_module
        from librarian.cli import app
        from librarian.storage.database import Database

        # Fabricate a populated v0.13 database on disk.
        db_file = tmp_path / "legacy.db"
        legacy = Database(db_path=str(db_file))
        zero = serialize_embedding([0.0] * 384)
        with legacy._connection() as conn:
            conn.execute(
                "INSERT INTO documents (path, title, content) VALUES (?, ?, ?)",
                ("/legacy/doc.md", "Legacy", "content"),
            )
            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (1, zero),
            )

        monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_file))

        result = CliRunner().invoke(app, ["search", "hello"])
        assert result.exit_code == 1
        assert "rebuild" in result.output.lower()

"""Regression tests for librarian.storage.database."""

from datetime import date, datetime
from pathlib import Path

import pytest

from librarian.storage.database import (
    Database,
    SqliteExtensionError,
    _load_sqlite_vec,
)
from librarian.types import AssetType, Document


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Fresh Database instance per test, isolated to a tmp file."""
    return Database(db_path=str(tmp_path / "test.db"))


class TestSqliteExtensionLoading:
    """Loading the sqlite-vec extension across SQLite builds with and without
    loadable-extension support."""

    def test_missing_enable_load_extension_raises_actionable_error(self) -> None:
        """A SQLite build without `enable_load_extension` must raise a clear
        SqliteExtensionError, not an opaque AttributeError. Regression: frozen
        PyInstaller binaries / macOS system Python crashed `cade mem add .`."""

        class _ExtensionlessConnection:
            """Mimics a stdlib sqlite3 build compiled without extension support."""

        with pytest.raises(SqliteExtensionError, match="loadable-extension support"):
            _load_sqlite_vec(_ExtensionlessConnection())  # type: ignore[arg-type]

    def test_load_succeeds_on_capable_connection(self, tmp_path: Path) -> None:
        """When the SQLite build supports extensions, sqlite-vec loads cleanly."""
        # A real Database init exercises _load_sqlite_vec; success means the
        # vec0 virtual table was created without error.
        database = Database(db_path=str(tmp_path / "vec.db"))
        stats = database.get_stats()
        assert stats["document_count"] == 0


class TestMetadataSerialization:
    """JSON-serialize document metadata containing types that YAML frontmatter
    emits but the stdlib json encoder doesn't know about."""

    def test_insert_with_date_metadata(self, db: Database) -> None:
        """A `date` in metadata (from `last_push: 2026-05-19` frontmatter) must
        not crash on insert. Regression: previously raised
        `TypeError: Object of type date is not JSON serializable`."""
        doc = Document(
            id=None,
            path="/note-with-date.md",
            title="Note",
            content="body",
            metadata={
                "last_push": date(2026, 5, 19),
                "updated_at": datetime(2026, 5, 19, 12, 30, 45),
                "tags": ["repo", "MOC"],
            },
            file_mtime=0.0,
            asset_type=AssetType.TEXT,
        )
        doc_id = db.insert_document(doc)
        assert doc_id is not None

        got = db.get_document_by_path("/note-with-date.md")
        assert got is not None
        # Dates round-trip as ISO strings; metadata is informational, not queried as dates.
        assert got.metadata["last_push"] == "2026-05-19"
        assert got.metadata["updated_at"].startswith("2026-05-19T12:30:45")
        assert got.metadata["tags"] == ["repo", "MOC"]

    def test_update_with_date_metadata(self, db: Database) -> None:
        """update_document() must also handle date metadata (parallel call site)."""
        doc = Document(
            id=None,
            path="/n.md",
            title="N",
            content="",
            metadata={"tag": "a"},
            file_mtime=0.0,
            asset_type=AssetType.TEXT,
        )
        db.insert_document(doc)

        loaded = db.get_document_by_path("/n.md")
        assert loaded is not None
        loaded.metadata = {"last_push": date(2026, 5, 19)}
        db.update_document(loaded)

        got = db.get_document_by_path("/n.md")
        assert got is not None
        assert got.metadata["last_push"] == "2026-05-19"

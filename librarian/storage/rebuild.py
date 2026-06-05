"""Database rebuild mechanics for the v0.13 -> v0.14 migration remedy.

The interactive ``libr index --rebuild`` command is a thin wrapper around these
substrate-level helpers: pick a non-colliding backup path, copy the existing
database aside, then wipe and recreate it empty under the current schema. The
mechanics live here (not in the CLI) so migration knowledge stays in the storage
layer; the CLI only owns confirmation prompts and source re-ingestion.
"""

import shutil
from datetime import datetime
from pathlib import Path

__all__ = ["backup_database", "backup_path_for", "recreate_empty"]


def backup_path_for(db_path: Path) -> Path:
    """Return a backup path for ``db_path`` that does not overwrite an existing one.

    The canonical name is ``<db>.v0-backup``. When it already exists (a prior
    rebuild took it), fall back to a timestamped sibling so the existing backup
    is preserved.
    """
    canonical = Path(str(db_path) + ".v0-backup")
    if not canonical.exists():
        return canonical
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"{db_path}.v0-backup-{stamp}")


def backup_database(db_path: Path) -> Path:
    """Copy ``db_path`` aside to a non-colliding backup and return its path.

    Raises (propagating the underlying ``OSError``) before the caller wipes the
    original, so a failed backup never proceeds to data loss.
    """
    backup = backup_path_for(db_path)
    shutil.copy2(db_path, backup)
    return backup


def recreate_empty(db_path: Path) -> None:
    """Delete ``db_path`` and recreate it empty under the current schema."""
    from librarian.storage import database as db_module
    from librarian.storage.database import get_database
    from librarian.storage.sqlite_storage import SQLiteStorage

    db_module._db_instance = None
    if db_path.exists():
        db_path.unlink()
    SQLiteStorage(database=get_database()).migrate()

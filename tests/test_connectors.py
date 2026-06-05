"""
Connector-contract tests.

These run with no database: a connector is a stateless, DB-free event generator,
so the contract can be verified entirely against a fixture filesystem and the
cursor it threads through ``fetch_changes``.
"""

from pathlib import Path

import pytest

from librarian import ids
from librarian.connectors import (
    DocumentSoftDelete,
    DocumentUpsert,
    LocalFileConnector,
)
from librarian.types import AssetType


async def _collect(connector: LocalFileConnector, state: dict) -> list:
    return [event async for event in connector.fetch_changes(state)]


def test_deterministic_ids_are_stable_across_calls() -> None:
    a = ids.document_id("local_file", "file", "/x/y.md")
    b = ids.document_id("local_file", "file", "/x/y.md")
    assert a == b
    assert len(a) == 64  # sha256 hex digest


def test_deterministic_ids_separate_components() -> None:
    # ("ab", "c") and ("a", "bc") must not collide.
    assert ids.chunk_id("ab", "c", "x") != ids.chunk_id("a", "bc", "x")
    assert ids.document_id("c", "ab", "x") != ids.document_id("c", "a", "bx")


async def test_local_file_connector_emits_upserts(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha content.")
    (tmp_path / "b.md").write_text("# B\n\nBeta content.")

    connector = LocalFileConnector([tmp_path])
    events = await _collect(connector, connector.initial_state())

    assert len(events) == 2
    assert all(isinstance(e, DocumentUpsert) for e in events)
    for e in events:
        assert e.source_type == "file"
        assert e.asset_type == AssetType.TEXT
        assert e.raw_content is not None  # textual asset carried inline
        assert e.document_source_uri and e.document_source_uri.startswith("file://")
        assert e.checkpoint is not None and "mtimes" in e.checkpoint


async def test_local_file_connector_is_idempotent_on_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha content.")
    connector = LocalFileConnector([tmp_path])

    first = await _collect(connector, connector.initial_state())
    assert len(first) == 1
    cursor = first[-1].checkpoint

    # Re-running from the advanced cursor yields nothing (file unchanged).
    second = await _collect(connector, cursor)
    assert second == []


async def test_local_file_connector_soft_deletes_removed_files(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("# A\n\nAlpha content.")
    connector = LocalFileConnector([tmp_path])

    first = await _collect(connector, connector.initial_state())
    cursor = first[-1].checkpoint

    f.unlink()
    events = await _collect(connector, cursor)

    assert len(events) == 1
    assert isinstance(events[0], DocumentSoftDelete)
    assert events[0].source_native_id == str(f.resolve())


async def test_local_file_connector_skips_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "note.md").write_text("# Note")
    connector = LocalFileConnector([tmp_path])

    events = await _collect(connector, connector.initial_state())
    paths = {Path(e.source_native_id).name for e in events}
    assert paths == {"note.md"}


@pytest.mark.parametrize("missing", ["does_not_exist.md"])
async def test_local_file_connector_handles_missing_path(tmp_path: Path, missing: str) -> None:
    connector = LocalFileConnector([tmp_path / missing])
    events = await _collect(connector, connector.initial_state())
    assert events == []


def _emitted_names(events: list) -> set[str]:
    return {Path(e.source_native_id).name for e in events}


async def test_local_file_connector_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.md\nbuild/\n")
    (tmp_path / "keep.md").write_text("# Keep")
    (tmp_path / "ignored.md").write_text("# Ignored")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.md").write_text("# Built")

    events = await _collect(LocalFileConnector([tmp_path]), {"mtimes": {}})
    assert _emitted_names(events) == {"keep.md"}


async def test_include_ignored_overrides_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.md\n")
    (tmp_path / "keep.md").write_text("# Keep")
    (tmp_path / "ignored.md").write_text("# Ignored")

    connector = LocalFileConnector([tmp_path], include_ignored=True)
    events = await _collect(connector, {"mtimes": {}})
    assert _emitted_names(events) == {"keep.md", "ignored.md"}


async def test_force_include_overrides_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.md\n")
    (tmp_path / "keep.md").write_text("# Keep")
    forced = tmp_path / "ignored.md"
    forced.write_text("# Ignored but forced")

    connector = LocalFileConnector([tmp_path], force_include=[str(forced)])
    events = await _collect(connector, {"mtimes": {}})
    assert _emitted_names(events) == {"keep.md", "ignored.md"}


async def test_librariantrack_overrides_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.md\n")
    (tmp_path / ".librariantrack").write_text("tracked.md\n")
    (tmp_path / "tracked.md").write_text("# Tracked")
    (tmp_path / "other.md").write_text("# Other")

    events = await _collect(LocalFileConnector([tmp_path]), {"mtimes": {}})
    assert _emitted_names(events) == {"tracked.md"}


async def test_skips_hidden_and_skip_dirs(tmp_path: Path) -> None:
    (tmp_path / "visible.md").write_text("# Visible")
    (tmp_path / ".hidden.md").write_text("# Hidden")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.md").write_text("# Dep")

    events = await _collect(LocalFileConnector([tmp_path]), {"mtimes": {}})
    assert _emitted_names(events) == {"visible.md"}

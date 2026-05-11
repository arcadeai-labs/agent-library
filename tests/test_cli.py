"""Tests for CLI behavior."""

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from librarian import cli


def test_add_directory_exits_nonzero_when_indexing_errors(tmp_path: Path, monkeypatch) -> None:
    """Directory add should fail automation when indexing reports errors."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    fixture = docs_dir / "fixture.md"
    fixture.write_text("# Synthetic Markdown Fixture\n\nContent.", encoding="utf-8")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "SOURCES_FILE", config_dir / "sources.json")
    monkeypatch.setattr(cli, "SETTINGS_FILE", config_dir / "settings.json")
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})

    async def fake_server_ingest(context: Any, directory: str) -> dict[str, Any]:
        return {
            "directory": directory,
            "total_files": 1,
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": [{"path": str(fixture), "error": "parser exploded"}],
            "files": [],
        }

    monkeypatch.setattr("librarian.server.index_directory_to_library", fake_server_ingest)

    result = CliRunner().invoke(cli.app, ["add", str(docs_dir), "--verbose"])

    assert result.exit_code == 1
    assert "Errors:" in result.output
    assert "parser exploded" in result.output

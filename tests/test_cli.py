"""Tests for CLI behavior."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from librarian import cli

LONG_WINDOWS_PATH = (
    r"C:\Users\example\Documents\Codex\2026-05-16"
    r"\concurso-browser-deckbuilder\docs\planning"
    r"\vgf-56-revolt-fx-distribution-and-ui-editor-plan.md"
)
LONG_WINDOWS_FILENAME = "vgf-56-revolt-fx-distribution-and-ui-editor-plan.md"


def assert_table_preserves_long_path_reference(result: Any) -> None:
    assert result.exit_code == 0
    assert "\ufffd" not in result.output
    assert "…" not in result.output
    normalized_output = "".join(
        char for char in result.output if char.isalnum() or char in "\\/:._-"
    )
    assert LONG_WINDOWS_FILENAME in normalized_output


def fake_document() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        title="VGF-56 plan",
        path=LONG_WINDOWS_PATH,
        asset_type=SimpleNamespace(value="text"),
    )


def test_add_directory_exits_nonzero_when_indexing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(cli, "console", Console(width=500, color_system=None))

    async def fake_server_ingest(
        context: Any,
        directory: str,
        include_ignored: bool = False,
        force_include: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "directory": directory,
            "total_files": 1,
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": [{"path": fixture.name, "error": "parser exploded"}],
            "files": [],
        }

    monkeypatch.setattr("librarian.server.index_directory_to_library", fake_server_ingest)

    # Force a wide terminal so Rich does not wrap the error message off-screen.
    monkeypatch.setenv("COLUMNS", "500")
    result = CliRunner().invoke(cli.app, ["add", str(docs_dir), "--verbose"])

    assert result.exit_code == 1
    assert "Errors:" in result.output
    assert "parser exploded" in result.output


def test_search_table_wraps_long_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narrow table output should wrap paths instead of replacing them with ellipses."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))

    async def fake_search_library(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "score": 1.0,
                "document_path": LONG_WINDOWS_PATH,
                "content": "matched content",
                "heading_path": None,
            }
        ]

    monkeypatch.setattr("librarian.server.search_library", fake_search_library)

    result = CliRunner().invoke(
        cli.app,
        [
            "search",
            "VGF-56 UI editor Fabric Tweakpane Pixi Layout Pixi UI",
            "--format",
            "table",
        ],
    )

    assert_table_preserves_long_path_reference(result)


def test_list_table_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source listing should not truncate long paths with an ellipsis."""
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        cli,
        "_load_sources",
        lambda: [{"name": "docs", "path": LONG_WINDOWS_PATH, "is_file": False}],
    )

    result = CliRunner().invoke(cli.app, ["list"])

    assert_table_preserves_long_path_reference(result)


def test_docs_overview_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document source overview should wrap long paths."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        cli,
        "_load_sources",
        lambda: [{"name": "docs", "path": LONG_WINDOWS_PATH, "is_file": False}],
    )
    monkeypatch.setattr(
        "librarian.storage.database.get_database",
        lambda: SimpleNamespace(list_documents=lambda: [fake_document()]),
    )

    result = CliRunner().invoke(cli.app, ["docs"])

    assert_table_preserves_long_path_reference(result)


def test_docs_list_table_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document listing should wrap long paths in table output."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        "librarian.storage.database.get_database",
        lambda: SimpleNamespace(list_documents=lambda: [fake_document()]),
    )

    result = CliRunner().invoke(cli.app, ["docs", "list", "--format", "table"])

    assert_table_preserves_long_path_reference(result)


def test_docs_search_table_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document title search should wrap long paths in table output."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        "librarian.storage.database.get_database",
        lambda: SimpleNamespace(list_documents=lambda: [fake_document()]),
    )

    result = CliRunner().invoke(cli.app, ["docs", "search", "VGF", "--format", "table"])

    assert_table_preserves_long_path_reference(result)


def test_search_paths_outputs_complete_long_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paths output remains the copyable full-path mode for search results."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})

    async def fake_search_library(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "score": 1.0,
                "document_path": LONG_WINDOWS_PATH,
                "content": "matched content",
                "heading_path": None,
            }
        ]

    monkeypatch.setattr("librarian.server.search_library", fake_search_library)

    result = CliRunner().invoke(
        cli.app,
        [
            "search",
            "VGF-56 UI editor Fabric Tweakpane Pixi Layout Pixi UI",
            "--format",
            "paths",
        ],
    )

    assert result.exit_code == 0
    assert result.output == f"{LONG_WINDOWS_PATH}\n"


def test_health_json_outputs_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health command should expose machine-readable diagnostics."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "_load_sources", lambda: [])

    fake_report = SimpleNamespace(
        to_dict=lambda: {
            "document_count": 1,
            "chunk_count": 2,
            "embedding_count": 2,
            "issues": [],
        }
    )
    monkeypatch.setattr("librarian.health.run_health_check", lambda **kwargs: fake_report)

    result = CliRunner().invoke(cli.app, ["health", "--json"])

    assert result.exit_code == 0
    assert '"document_count": 1' in result.output
    assert '"chunk_count": 2' in result.output

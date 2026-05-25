"""Tests for .gitignore-aware skip logic."""

from __future__ import annotations

from pathlib import Path

from librarian.sources.ignore import (
    ALWAYS_SKIP_DIRS,
    GitignoreMatcher,
    LibrarianTrackMatcher,
    normalize_force_include,
    should_skip_file,
)

SUPPORTED = {".md", ".py", ".txt"}


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestGitignoreMatcher:
    def test_no_gitignore_means_nothing_is_ignored(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "a.md", "hi")
        matcher = GitignoreMatcher(tmp_path)
        assert not matcher.is_ignored(f)

    def test_root_gitignore_excludes_listed_files(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "build/\n*.log\n")
        log = _write(tmp_path / "app.log", "x")
        built = _write(tmp_path / "build" / "out.md", "x")
        kept = _write(tmp_path / "keep.md", "x")

        matcher = GitignoreMatcher(tmp_path)
        assert matcher.is_ignored(log)
        assert matcher.is_ignored(built)
        assert not matcher.is_ignored(kept)

    def test_floating_pattern_matches_at_any_depth(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "node_modules\n")
        nested = _write(tmp_path / "pkg" / "node_modules" / "lib.js", "x")
        matcher = GitignoreMatcher(tmp_path)
        assert matcher.is_ignored(nested)

    def test_anchored_pattern_only_matches_at_root(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "/foo.md\n")
        at_root = _write(tmp_path / "foo.md", "x")
        nested = _write(tmp_path / "sub" / "foo.md", "x")
        matcher = GitignoreMatcher(tmp_path)
        assert matcher.is_ignored(at_root)
        assert not matcher.is_ignored(nested)

    def test_nested_gitignore_is_scoped_to_its_directory(self, tmp_path: Path) -> None:
        _write(tmp_path / "sub" / ".gitignore", "secret.md\n")
        nested_secret = _write(tmp_path / "sub" / "secret.md", "x")
        elsewhere_secret = _write(tmp_path / "other" / "secret.md", "x")

        matcher = GitignoreMatcher(tmp_path)
        assert matcher.is_ignored(nested_secret)
        assert not matcher.is_ignored(elsewhere_secret)

    def test_negation_unignores_a_specific_file(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "*.md\n!keep.md\n")
        ignored = _write(tmp_path / "drop.md", "x")
        kept = _write(tmp_path / "keep.md", "x")

        matcher = GitignoreMatcher(tmp_path)
        assert matcher.is_ignored(ignored)
        assert not matcher.is_ignored(kept)

    def test_paths_outside_root_are_not_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        _write(root / ".gitignore", "*.md\n")
        outside = _write(tmp_path / "outside.md", "x")
        matcher = GitignoreMatcher(root)
        assert not matcher.is_ignored(outside)


class TestShouldSkipFile:
    def test_always_skips_node_modules_even_without_gitignore(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "proj" / "node_modules" / "lib" / "a.md", "x")
        assert should_skip_file(f, SUPPORTED, gitignore_matcher=None)

    def test_always_skip_dirs_contains_expected(self) -> None:
        assert "node_modules" in ALWAYS_SKIP_DIRS
        assert ".git" in ALWAYS_SKIP_DIRS
        assert "__pycache__" in ALWAYS_SKIP_DIRS

    def test_unsupported_extension_skipped(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "image.png", "")
        assert should_skip_file(f, SUPPORTED, gitignore_matcher=None)

    def test_supported_file_kept_without_matcher(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "doc.md", "hi")
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=None)

    def test_gitignore_match_causes_skip(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "drafts/\n")
        f = _write(tmp_path / "drafts" / "wip.md", "x")
        matcher = GitignoreMatcher(tmp_path)
        assert should_skip_file(f, SUPPORTED, gitignore_matcher=matcher)

    def test_include_ignored_means_no_matcher_passed(self, tmp_path: Path) -> None:
        """When --include-ignored is set, callers pass matcher=None and the
        file survives the skip check (assuming it is otherwise valid)."""
        _write(tmp_path / ".gitignore", "drafts/\n")
        f = _write(tmp_path / "drafts" / "wip.md", "x")
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=None)


class TestForceInclude:
    def test_force_include_directory_overrides_gitignore(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "drafts/\n")
        f = _write(tmp_path / "drafts" / "wip.md", "x")
        matcher = GitignoreMatcher(tmp_path)
        forced = normalize_force_include([str(tmp_path / "drafts")])

        assert should_skip_file(f, SUPPORTED, gitignore_matcher=matcher)
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=matcher, force_include=forced)

    def test_force_include_file_overrides_gitignore(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "*.md\n")
        f = _write(tmp_path / "keep.md", "x")
        matcher = GitignoreMatcher(tmp_path)
        forced = normalize_force_include([str(f)])

        assert should_skip_file(f, SUPPORTED, gitignore_matcher=matcher)
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=matcher, force_include=forced)

    def test_force_include_overrides_always_skip_dirs(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "proj" / "node_modules" / "pkg" / "lib.py", "x")
        forced = normalize_force_include([str(tmp_path / "proj" / "node_modules" / "pkg")])

        assert should_skip_file(f, SUPPORTED, gitignore_matcher=None)
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=None, force_include=forced)

    def test_force_include_does_not_rescue_unsupported_extension(self, tmp_path: Path) -> None:
        """Force-include bypasses skip rules, but unparseable file types still
        cannot enter the index — the parser registry has no parser for them."""
        f = _write(tmp_path / "binary.exe", "x")
        forced = normalize_force_include([str(tmp_path)])
        assert should_skip_file(f, SUPPORTED, gitignore_matcher=None, force_include=forced)

    def test_force_include_does_not_rescue_hidden_files(self, tmp_path: Path) -> None:
        f = _write(tmp_path / ".hidden.md", "x")
        forced = normalize_force_include([str(tmp_path)])
        assert should_skip_file(f, SUPPORTED, gitignore_matcher=None, force_include=forced)

    def test_normalize_drops_nonexistent_paths(self, tmp_path: Path) -> None:
        present = _write(tmp_path / "exists.md", "x")
        forced = normalize_force_include([
            str(present),
            str(tmp_path / "does_not_exist"),
        ])
        assert present.resolve() in forced
        assert len(forced) == 1

    def test_normalize_handles_none_and_empty(self) -> None:
        assert normalize_force_include(None) == frozenset()
        assert normalize_force_include([]) == frozenset()


class TestLibrarianTrackMatcher:
    def test_no_trackfile_means_nothing_is_tracked(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "a.md", "x")
        matcher = LibrarianTrackMatcher(tmp_path)
        assert not matcher.is_tracked(f)

    def test_root_trackfile_unignores_listed_files(self, tmp_path: Path) -> None:
        _write(tmp_path / ".librariantrack", "drafts/\n")
        kept = _write(tmp_path / "drafts" / "wip.md", "x")
        other = _write(tmp_path / "other.md", "x")
        matcher = LibrarianTrackMatcher(tmp_path)
        assert matcher.is_tracked(kept)
        assert not matcher.is_tracked(other)

    def test_track_overrides_gitignore_for_matched_files(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "drafts/\n")
        _write(tmp_path / ".librariantrack", "drafts/\n")
        f = _write(tmp_path / "drafts" / "wip.md", "x")

        gitignore = GitignoreMatcher(tmp_path)
        track = LibrarianTrackMatcher(tmp_path)

        assert should_skip_file(f, SUPPORTED, gitignore_matcher=gitignore)
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=gitignore, track_matcher=track)

    def test_track_overrides_always_skip_dirs(self, tmp_path: Path) -> None:
        _write(tmp_path / ".librariantrack", "node_modules/pkg/\n")
        f = _write(tmp_path / "node_modules" / "pkg" / "lib.py", "x")
        track = LibrarianTrackMatcher(tmp_path)

        assert should_skip_file(f, SUPPORTED, gitignore_matcher=None)
        assert not should_skip_file(f, SUPPORTED, gitignore_matcher=None, track_matcher=track)

    def test_nested_trackfile_is_scoped_to_its_directory(self, tmp_path: Path) -> None:
        _write(tmp_path / ".gitignore", "*.md\n")
        _write(tmp_path / "kept" / ".librariantrack", "*.md\n")
        kept = _write(tmp_path / "kept" / "doc.md", "x")
        elsewhere = _write(tmp_path / "other" / "doc.md", "x")

        gitignore = GitignoreMatcher(tmp_path)
        track = LibrarianTrackMatcher(tmp_path)

        assert not should_skip_file(
            kept, SUPPORTED, gitignore_matcher=gitignore, track_matcher=track
        )
        assert should_skip_file(
            elsewhere, SUPPORTED, gitignore_matcher=gitignore, track_matcher=track
        )

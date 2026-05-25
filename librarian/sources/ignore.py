"""File/directory skip logic for indexing.

Three layers, in order of precedence (highest wins):
1. Force-include set (--force-include) and `.librariantrack` patterns — make a
   file survive every skip rule below, including the skip-dirs baseline and any
   `.gitignore` match.
2. `.gitignore` aggregation under the source root (`GitignoreMatcher`).
3. Hardcoded but config-overridable defaults: directory names from
   `INDEX_SKIP_DIRS`, binary/archive extensions from `INDEX_SKIP_EXTENSIONS`,
   hidden files, and unsupported extensions.
"""

from __future__ import annotations

from pathlib import Path

from pathspec import GitIgnoreSpec

from librarian.config import INDEX_SKIP_DIRS, INDEX_SKIP_EXTENSIONS

# Re-exported under the historical names so callers and tests that imported
# them from this module keep working.
ALWAYS_SKIP_DIRS = INDEX_SKIP_DIRS
SKIP_EXTENSIONS = INDEX_SKIP_EXTENSIONS

# Filename used for per-directory force-include patterns. Patterns inside a
# `.librariantrack` file behave like inverse `.gitignore` patterns: anything
# matched is indexed even if it would otherwise be skipped.
LIBRARIANTRACK_FILENAME = ".librariantrack"


class GitignoreMatcher:
    """Tells whether a file is excluded by any `.gitignore` under a root.

    Aggregates patterns from every `.gitignore` found at or below the root.
    Patterns from nested files are anchored to their containing directory,
    mirroring git's own semantics for the common cases (anchored patterns,
    floating patterns, directory-only patterns, and negations).
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._spec: GitIgnoreSpec | None = self._build_spec()

    def _build_spec(self) -> GitIgnoreSpec | None:
        if not self.root.is_dir():
            return None
        lines = _collect_anchored_patterns(self.root, ".gitignore")
        if not lines:
            return None
        return GitIgnoreSpec.from_lines(lines)

    def is_ignored(self, file_path: Path) -> bool:
        if self._spec is None:
            return False
        try:
            rel = file_path.resolve().relative_to(self.root)
        except ValueError:
            return False
        return self._spec.match_file(rel.as_posix())


class LibrarianTrackMatcher:
    """Tells whether a file is force-included by a `.librariantrack` file.

    `.librariantrack` patterns work like `.gitignore` patterns but inverted:
    a match means "track this file no matter what other skip rules say".
    Patterns are anchored to the containing directory the same way gitignore
    patterns are.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._spec: GitIgnoreSpec | None = self._build_spec()

    def _build_spec(self) -> GitIgnoreSpec | None:
        if not self.root.is_dir():
            return None
        lines = _collect_anchored_patterns(self.root, LIBRARIANTRACK_FILENAME)
        if not lines:
            return None
        return GitIgnoreSpec.from_lines(lines)

    def is_tracked(self, file_path: Path) -> bool:
        if self._spec is None:
            return False
        try:
            rel = file_path.resolve().relative_to(self.root)
        except ValueError:
            return False
        return self._spec.match_file(rel.as_posix())


def _collect_anchored_patterns(root: Path, filename: str) -> list[str]:
    """Aggregate patterns from every file named `filename` under `root`.

    Outer files come before inner ones so later (deeper) lines win, matching
    git's "deeper file overrides" semantics.
    """
    lines: list[str] = []
    files = sorted(root.rglob(filename), key=lambda p: len(p.parts))
    for f in files:
        try:
            rel_dir = f.parent.resolve().relative_to(root)
        except ValueError:
            continue
        prefix = "" if rel_dir == Path(".") else f"{rel_dir.as_posix()}/"
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for raw in content.splitlines():
            pattern = _prefix_pattern(raw, prefix)
            if pattern is not None:
                lines.append(pattern)
    return lines


def _prefix_pattern(raw: str, prefix: str) -> str | None:
    """Translate a single gitignore-style line to be anchored under `prefix`.

    Returns None for blank lines and comments. `prefix` is the file's
    directory relative to the matcher root, with a trailing slash (or empty
    string when the file lives at the root).
    """
    line = raw.rstrip()
    if not line or line.startswith("#"):
        return None

    negate = line.startswith("!")
    if negate:
        line = line[1:]

    if not prefix:
        return ("!" + line) if negate else line

    # Determine whether the pattern is anchored to its directory.
    # Per git: a leading '/' or any '/' before the end of the pattern anchors
    # it; otherwise the pattern matches at any depth under that directory.
    stripped = line.rstrip("/")
    if line.startswith("/"):
        new = f"/{prefix}{line[1:]}"
    elif "/" in stripped:
        new = f"/{prefix}{line}"
    else:
        new = f"{prefix}**/{line}"
    return ("!" + new) if negate else new


def _is_force_included(
    file_path: Path,
    force_include: frozenset[Path] | None,
    track_matcher: LibrarianTrackMatcher | None,
) -> bool:
    """A force-include match overrides every skip rule except unparseable types."""
    if force_include:
        resolved = file_path.resolve()
        for forced in force_include:
            try:
                resolved.relative_to(forced)
            except ValueError:
                continue
            return True
    return track_matcher is not None and track_matcher.is_tracked(file_path)


def normalize_force_include(paths: list[str] | None) -> frozenset[Path]:
    """Resolve a list of user-supplied force-include paths.

    Non-existent paths are dropped silently — they cannot match anything, and
    surfacing them as errors would force callers to validate before us.
    """
    if not paths:
        return frozenset()
    out: set[Path] = set()
    for raw in paths:
        try:
            p = Path(raw).expanduser().resolve()
        except OSError:
            continue
        if p.exists():
            out.add(p)
    return frozenset(out)


def should_skip_file(
    file_path: Path,
    supported_extensions: set[str],
    gitignore_matcher: GitignoreMatcher | None = None,
    force_include: frozenset[Path] | None = None,
    track_matcher: LibrarianTrackMatcher | None = None,
) -> bool:
    """Decide whether a file should be skipped during indexing.

    Force-include (via `force_include` paths or a `.librariantrack` match)
    bypasses the skip-dirs baseline and any `.gitignore` rule. It does not
    bypass unparseable file types (unsupported or binary extensions, hidden
    files, files without an extension) because the indexer can't process them.
    """
    forced = _is_force_included(file_path, force_include, track_matcher)

    if not forced:
        for parent in file_path.parents:
            if parent.name in INDEX_SKIP_DIRS:
                return True

    if file_path.name.startswith("."):
        return True

    if file_path.suffix.lower() in INDEX_SKIP_EXTENSIONS:
        return True

    if not file_path.suffix:
        return True

    if file_path.suffix.lower() not in supported_extensions:
        return True

    if forced:
        return False

    return gitignore_matcher is not None and gitignore_matcher.is_ignored(file_path)

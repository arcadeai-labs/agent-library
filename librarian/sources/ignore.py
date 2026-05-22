"""File/directory skip logic for indexing, including .gitignore support."""

from __future__ import annotations

from pathlib import Path

from pathspec import GitIgnoreSpec

# Directories that are always skipped, even when --include-ignored is set.
# These are caches, VCS metadata, and OS junk that should never enter the library.
ALWAYS_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__MACOSX",
    ".DS_Store",
})

# Binary / system file extensions we never index.
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    # Executables and binaries
    ".exe",
    ".bin",
    ".dll",
    ".so",
    ".dylib",
    ".a",
    ".o",
    # Disk images and archives
    ".dmg",
    ".iso",
    ".img",
    ".app",
    ".pkg",
    # Compressed archives
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    # Python compiled
    ".pyc",
    ".pyo",
    ".pyd",
    # System files
    ".lock",
    ".log",
    ".tmp",
    ".temp",
    ".cache",
    # Media files (large binaries)
    # TODO: revisit once audio/video parsers land — these will need to move
    # out of the skip list and into the parser registry as new asset types.
    ".mp4",
    ".mp3",
    ".wav",
    ".avi",
    ".mov",
    ".flac",
    # Font files
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
})


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

        lines: list[str] = []
        # Sort so outer .gitignore files come before inner ones; later lines
        # win in GitIgnoreSpec, which matches git's "deeper file overrides".
        gitignores = sorted(
            self.root.rglob(".gitignore"),
            key=lambda p: len(p.parts),
        )
        for gitignore in gitignores:
            try:
                rel_dir = gitignore.parent.resolve().relative_to(self.root)
            except ValueError:
                continue
            prefix = "" if rel_dir == Path(".") else f"{rel_dir.as_posix()}/"
            try:
                content = gitignore.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for raw in content.splitlines():
                pattern = _prefix_pattern(raw, prefix)
                if pattern is not None:
                    lines.append(pattern)

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


def _prefix_pattern(raw: str, prefix: str) -> str | None:
    """Translate a single .gitignore line to be anchored under `prefix`.

    Returns None for blank lines and comments. `prefix` is the gitignore's
    directory relative to the matcher root, with a trailing slash (or empty
    string when the gitignore lives at the root).
    """
    line = raw.rstrip()
    if not line or line.startswith("#"):
        return None

    negate = line.startswith("!")
    if negate:
        line = line[1:]

    if not prefix:
        return ("!" + line) if negate else line

    # Determine whether the pattern is anchored to its gitignore's directory.
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


def should_skip_file(
    file_path: Path,
    supported_extensions: set[str],
    gitignore_matcher: GitignoreMatcher | None = None,
) -> bool:
    """Decide whether a file should be skipped during indexing.

    Hardcoded baseline (always applied): cache/VCS directories, binary
    extensions, hidden files, and files lacking a supported extension.
    When `gitignore_matcher` is provided, files it marks as ignored are
    also skipped.
    """
    for parent in file_path.parents:
        if parent.name in ALWAYS_SKIP_DIRS:
            return True

    if file_path.name.startswith("."):
        return True

    if file_path.suffix.lower() in SKIP_EXTENSIONS:
        return True

    if not file_path.suffix:
        return True

    if file_path.suffix.lower() not in supported_extensions:
        return True

    return gitignore_matcher is not None and gitignore_matcher.is_ignored(file_path)

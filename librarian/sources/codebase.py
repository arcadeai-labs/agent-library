"""
Codebase source management.

Handles detection, configuration, and indexing of code repositories.
"""

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from librarian.config import (
    CODEBASE_AUTO_DETECT,
    CODEBASE_INDEX_TESTS,
    CODEBASE_MAX_FILE_SIZE_KB,
    SOURCES_CONFIG_PATH,
)
from librarian.sources.ignore import IgnorePatterns
from librarian.types import CodebaseMetadata, ProgrammingLanguage, SourceConfig, SourceType


def detect_language(file_path: Path) -> ProgrammingLanguage | None:
    """
    Detect programming language from file extension.

    Args:
        file_path: Path to source file.

    Returns:
        Programming language enum, or None if not recognized.
    """
    extension_map = {
        ".py": ProgrammingLanguage.PYTHON,
        ".js": ProgrammingLanguage.JAVASCRIPT,
        ".ts": ProgrammingLanguage.TYPESCRIPT,
        ".jsx": ProgrammingLanguage.JAVASCRIPT,
        ".tsx": ProgrammingLanguage.TYPESCRIPT,
        ".go": ProgrammingLanguage.GO,
        ".rs": ProgrammingLanguage.RUST,
        ".java": ProgrammingLanguage.JAVA,
        ".cpp": ProgrammingLanguage.CPP,
        ".cc": ProgrammingLanguage.CPP,
        ".cxx": ProgrammingLanguage.CPP,
        ".c": ProgrammingLanguage.C,
        ".h": ProgrammingLanguage.C,
        ".hpp": ProgrammingLanguage.CPP,
        ".rb": ProgrammingLanguage.RUBY,
        ".php": ProgrammingLanguage.PHP,
        ".swift": ProgrammingLanguage.SWIFT,
        ".kt": ProgrammingLanguage.KOTLIN,
        ".cs": ProgrammingLanguage.CSHARP,
        ".html": ProgrammingLanguage.HTML,
        ".css": ProgrammingLanguage.CSS,
        ".sql": ProgrammingLanguage.SQL,
        ".sh": ProgrammingLanguage.SHELL,
        ".bash": ProgrammingLanguage.SHELL,
        ".zsh": ProgrammingLanguage.SHELL,
    }
    extension = file_path.suffix.lower()
    return extension_map.get(extension)


def detect_primary_language(codebase_path: Path) -> ProgrammingLanguage | None:
    """
    Detect the primary programming language in a codebase.

    Args:
        codebase_path: Path to codebase root.

    Returns:
        Primary programming language, or None if no code files found.
    """
    language_counts: Counter[ProgrammingLanguage] = Counter()

    # Scan directory for code files
    ignore_patterns = IgnorePatterns(codebase_path)

    for file_path in codebase_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip ignored files
        if ignore_patterns.should_ignore(file_path):
            continue

        # Detect language
        language = detect_language(file_path)
        if language and language != ProgrammingLanguage.OTHER:
            language_counts[language] += 1

    # Return most common language
    if language_counts:
        return language_counts.most_common(1)[0][0]
    return None


def detect_framework(codebase_path: Path, language: ProgrammingLanguage | None) -> str | None:
    """
    Detect framework or build system based on config files.

    Args:
        codebase_path: Path to codebase root.
        language: Primary programming language.

    Returns:
        Framework name, or None if not detected.
    """
    # Check for common framework indicators
    framework_indicators = {
        "package.json": "node",
        "requirements.txt": "python",
        "Pipfile": "python-pipenv",
        "pyproject.toml": "python-poetry",
        "setup.py": "python-setuptools",
        "Cargo.toml": "rust-cargo",
        "go.mod": "go-modules",
        "pom.xml": "java-maven",
        "build.gradle": "java-gradle",
        "Gemfile": "ruby-bundler",
        "composer.json": "php-composer",
        "Package.swift": "swift-spm",
    }

    for indicator, framework in framework_indicators.items():
        if (codebase_path / indicator).exists():
            return framework

    # Language-specific defaults
    if language == ProgrammingLanguage.JAVASCRIPT:
        return "javascript"
    elif language == ProgrammingLanguage.TYPESCRIPT:
        return "typescript"
    elif language == ProgrammingLanguage.PYTHON:
        return "python"

    return None


def get_git_remote(codebase_path: Path) -> str | None:
    """
    Get git remote URL for a codebase.

    Args:
        codebase_path: Path to codebase root.

    Returns:
        Git remote URL, or None if not a git repo.
    """
    git_dir = codebase_path / ".git"
    if not git_dir.exists():
        return None

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],  # noqa: S607
            cwd=codebase_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def get_languages_in_codebase(codebase_path: Path) -> list[ProgrammingLanguage]:
    """
    Get all programming languages used in a codebase.

    Args:
        codebase_path: Path to codebase root.

    Returns:
        List of programming languages found.
    """
    languages: set[ProgrammingLanguage] = set()
    ignore_patterns = IgnorePatterns(codebase_path)

    for file_path in codebase_path.rglob("*"):
        if not file_path.is_file():
            continue

        if ignore_patterns.should_ignore(file_path):
            continue

        language = detect_language(file_path)
        if language and language != ProgrammingLanguage.OTHER:
            languages.add(language)

    return sorted(languages, key=lambda x: x.value)


class CodebaseManager:
    """Manages codebase sources and their configuration."""

    def __init__(self, sources_config_path: str = SOURCES_CONFIG_PATH) -> None:
        """
        Initialize codebase manager.

        Args:
            sources_config_path: Path to sources configuration file.
        """
        self.sources_config_path = Path(sources_config_path)
        self.sources: list[SourceConfig] = []
        self._load_sources()

    def _load_sources(self) -> None:
        """Load sources from configuration file."""
        if not self.sources_config_path.exists():
            return

        try:
            with open(self.sources_config_path, encoding="utf-8") as f:
                data = json.load(f)

            for source_data in data.get("sources", []):
                # Parse codebase metadata if present
                codebase_metadata = None
                if source_data.get("codebase_metadata"):
                    meta = source_data["codebase_metadata"]
                    codebase_metadata = CodebaseMetadata(
                        git_remote=meta.get("git_remote"),
                        primary_language=(
                            ProgrammingLanguage(meta["primary_language"])
                            if meta.get("primary_language")
                            else None
                        ),
                        languages=[ProgrammingLanguage(lang) for lang in meta.get("languages", [])],
                        framework=meta.get("framework"),
                        dependencies=meta.get("dependencies", {}),
                        exclude_patterns=meta.get("exclude_patterns", []),
                        include_patterns=meta.get("include_patterns", []),
                        max_file_size_kb=meta.get("max_file_size_kb", CODEBASE_MAX_FILE_SIZE_KB),
                    )

                source = SourceConfig(
                    name=source_data["name"],
                    path=source_data["path"],
                    source_type=SourceType(source_data.get("source_type", "documents")),
                    recursive=source_data.get("recursive", True),
                    codebase_metadata=codebase_metadata,
                )
                self.sources.append(source)

        except (json.JSONDecodeError, KeyError) as e:
            # Log error but don't fail
            import logging

            logging.warning(f"Failed to load sources config: {e}")

    def _save_sources(self) -> None:
        """Save sources to configuration file."""
        # Ensure directory exists
        self.sources_config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        sources_data = []
        for source in self.sources:
            source_dict: dict[str, Any] = {
                "name": source.name,
                "path": source.path,
                "source_type": source.source_type.value,
                "recursive": source.recursive,
            }

            if source.codebase_metadata:
                meta = source.codebase_metadata
                source_dict["codebase_metadata"] = {
                    "git_remote": meta.git_remote,
                    "primary_language": meta.primary_language.value
                    if meta.primary_language
                    else None,
                    "languages": [lang.value for lang in meta.languages],
                    "framework": meta.framework,
                    "dependencies": meta.dependencies,
                    "exclude_patterns": meta.exclude_patterns,
                    "include_patterns": meta.include_patterns,
                    "max_file_size_kb": meta.max_file_size_kb,
                }

            sources_data.append(source_dict)

        data = {"sources": sources_data}

        with open(self.sources_config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_codebase(
        self,
        name: str,
        path: Path | str,
        auto_detect: bool = CODEBASE_AUTO_DETECT,
        index_tests: bool = CODEBASE_INDEX_TESTS,
    ) -> SourceConfig:
        """
        Add a codebase source.

        Args:
            name: Human-readable name for the codebase.
            path: Path to codebase root directory.
            auto_detect: Whether to auto-detect language and framework.
            index_tests: Whether to include test files.

        Returns:
            SourceConfig for the added codebase.
        """
        codebase_path = Path(path).resolve()

        if not codebase_path.is_dir():
            raise ValueError(f"Path is not a directory: {codebase_path}")

        # Auto-detect metadata
        codebase_metadata = None
        if auto_detect:
            primary_language = detect_primary_language(codebase_path)
            languages = get_languages_in_codebase(codebase_path)
            framework = detect_framework(codebase_path, primary_language)
            git_remote = get_git_remote(codebase_path)

            # Default exclude patterns
            exclude_patterns = []
            if not index_tests:
                exclude_patterns.extend(["**/test/**", "**/tests/**", "**/*_test.*", "**/test_*"])

            codebase_metadata = CodebaseMetadata(
                git_remote=git_remote,
                primary_language=primary_language,
                languages=languages,
                framework=framework,
                exclude_patterns=exclude_patterns,
                max_file_size_kb=CODEBASE_MAX_FILE_SIZE_KB,
            )

        # Create source config
        source = SourceConfig(
            name=name,
            path=str(codebase_path),
            source_type=SourceType.CODEBASE,
            recursive=True,
            codebase_metadata=codebase_metadata,
        )

        # Add to sources and save
        self.sources.append(source)
        self._save_sources()

        return source

    def get_codebase(self, name: str) -> SourceConfig | None:
        """
        Get a codebase by name.

        Args:
            name: Codebase name.

        Returns:
            SourceConfig if found, None otherwise.
        """
        for source in self.sources:
            if source.name == name and source.source_type == SourceType.CODEBASE:
                return source
        return None

    def list_codebases(self) -> list[SourceConfig]:
        """
        List all codebase sources.

        Returns:
            List of codebase source configs.
        """
        return [s for s in self.sources if s.source_type == SourceType.CODEBASE]

    def remove_codebase(self, name: str) -> bool:
        """
        Remove a codebase source.

        Args:
            name: Codebase name.

        Returns:
            True if removed, False if not found.
        """
        for i, source in enumerate(self.sources):
            if source.name == name and source.source_type == SourceType.CODEBASE:
                self.sources.pop(i)
                self._save_sources()
                return True
        return False

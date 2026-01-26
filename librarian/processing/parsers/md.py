"""
Markdown parser implementation.

Parses standard markdown files, extracting frontmatter metadata,
section structure, and content organization.
"""

import logging
import re
from pathlib import Path
from typing import Any

import frontmatter
import yaml

from librarian.processing.parsers.base import BaseParser
from librarian.types import AssetType, ParsedDocument, Section

logger = logging.getLogger(__name__)


class MarkdownParser(BaseParser):
    """
    Parser for markdown documents.

    Extracts:
    - YAML frontmatter metadata
    - Section hierarchy based on headers
    - Title from frontmatter, first H1, or filename
    """

    # Match markdown headers (# through ######)
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @property
    def supported_extensions(self) -> set[str]:
        """Return supported markdown extensions."""
        return {".md", ".markdown", ".mdown", ".mkd"}

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            ParsedDocument with extracted metadata and structure.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        try:
            raw_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for non-UTF-8 files
            raw_content = path.read_text(encoding="latin-1")

        return self.parse_content(raw_content, str(path))

    def parse_content(self, content: str, path: str = "") -> ParsedDocument:
        """
        Parse markdown content from a string.

        Args:
            content: Markdown content string.
            path: Optional source path for reference.

        Returns:
            ParsedDocument with structure and metadata.
        """
        metadata, body = self._extract_frontmatter(content)
        title = self._extract_title(metadata, body, path)
        sections = self._parse_sections(body)

        return ParsedDocument(
            path=path,
            title=title,
            content=body,
            metadata=metadata,
            sections=sections,
            raw_content=content,
            asset_type=AssetType.TEXT,
        )

    def _extract_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """
        Extract YAML frontmatter from markdown.

        Tries python-frontmatter first, falls back to manual extraction.

        Returns:
            Tuple of (metadata dict, body without frontmatter).
        """
        try:
            post = frontmatter.loads(content)
            return dict(post.metadata), post.content
        except Exception as e:
            logger.debug("Frontmatter library failed: %s", e)
            return self._manual_frontmatter_extract(content)

    def _manual_frontmatter_extract(self, content: str) -> tuple[dict[str, Any], str]:
        """
        Manually extract frontmatter when library fails.

        Handles edge cases like malformed YAML or unusual delimiters.
        """
        if not content.startswith("---"):
            return {}, content

        lines = content.split("\n")
        end_idx = None

        # Find closing delimiter
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = i
                break

        if end_idx is None:
            return {}, content

        fm_text = "\n".join(lines[1:end_idx])
        body = "\n".join(lines[end_idx + 1 :])

        try:
            metadata = yaml.safe_load(fm_text) or {}
            if not isinstance(metadata, dict):
                metadata = {}
        except yaml.YAMLError:
            metadata = {}

        return metadata, body

    def _extract_title(self, metadata: dict[str, Any], content: str, path: str) -> str | None:
        """
        Extract document title with fallback chain.

        Priority:
        1. Frontmatter 'title' field
        2. First H1 header in content
        3. Filename stem
        """
        # Check frontmatter
        if metadata.get("title"):
            return str(metadata["title"]).strip()

        # Check for first H1
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fallback to filename
        if path:
            return Path(path).stem

        return None

    def _parse_sections(self, content: str) -> list[Section]:
        """
        Parse content into sections based on headers.

        Each header creates a new section containing content until
        the next header or end of document.
        """
        sections: list[Section] = []
        headers = list(self.HEADER_PATTERN.finditer(content))

        if not headers:
            # No headers - treat entire content as one section
            if content.strip():
                sections.append(
                    Section(
                        title="",
                        level=0,
                        content=content,
                        start_pos=0,
                        end_pos=len(content),
                    )
                )
            return sections

        for i, match in enumerate(headers):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()

            # End at next header or document end
            end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            # Content excludes the header line
            section_content = content[match.end() : end_pos].strip()

            sections.append(
                Section(
                    title=title,
                    level=level,
                    content=section_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

        return sections

    def get_section_path(self, sections: list[Section], target: Section) -> str:
        """
        Build hierarchical path for a section.

        Creates a path like "Chapter 1 > Section 2 > Subsection"
        based on header levels.

        Args:
            sections: All sections in the document.
            target: The target section.

        Returns:
            Heading path string.
        """
        path_parts: list[str] = []
        current_level = 0

        for section in sections:
            if section.start_pos > target.start_pos:
                break

            if section.level > current_level or section == target:
                if section.title:
                    # Remove ancestors at same or lower level
                    while (
                        path_parts
                        and len(path_parts) > 0
                        and sections[len(path_parts) - 1].level >= section.level
                    ):
                        path_parts.pop()
                    path_parts.append(section.title)
                current_level = section.level

        return " > ".join(path_parts) if path_parts else ""

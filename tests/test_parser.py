"""Tests for markdown parsing functionality."""

import pytest

from librarian.processing.parsers.md import MarkdownParser


@pytest.fixture
def parser() -> MarkdownParser:
    """Create a markdown parser instance."""
    return MarkdownParser()


class TestMarkdownParser:
    """Tests for the MarkdownParser class."""

    def test_parse_simple_content(self, parser: MarkdownParser) -> None:
        """Test parsing simple markdown content."""
        content = "# Title\n\nSome content here."
        result = parser.parse_content(content, "/path/test.md")

        assert result.title == "Title"
        assert "Some content here" in result.content
        assert len(result.sections) >= 1

    def test_parse_frontmatter(self, parser: MarkdownParser) -> None:
        """Test parsing YAML frontmatter."""
        content = """---
title: Frontmatter Title
tags:
  - tag1
  - tag2
---

# Header

Content here."""
        result = parser.parse_content(content, "/path/test.md")

        assert result.title == "Frontmatter Title"
        assert result.metadata.get("tags") == ["tag1", "tag2"]

    def test_parse_no_frontmatter(self, parser: MarkdownParser) -> None:
        """Test parsing content without frontmatter."""
        content = "# Just a Header\n\nNo frontmatter here."
        result = parser.parse_content(content, "/path/test.md")

        assert result.title == "Just a Header"
        assert result.metadata == {}

    def test_extract_title_priority(self, parser: MarkdownParser) -> None:
        """Test title extraction priority: frontmatter > h1 > filename."""
        # Frontmatter takes priority
        content1 = "---\ntitle: FM Title\n---\n# Header Title"
        result1 = parser.parse_content(content1, "/path/file.md")
        assert result1.title == "FM Title"

        # H1 if no frontmatter title
        content2 = "---\ntags: [test]\n---\n# Header Title"
        result2 = parser.parse_content(content2, "/path/file.md")
        assert result2.title == "Header Title"

        # Filename as fallback
        content3 = "No title anywhere"
        result3 = parser.parse_content(content3, "/path/fallback.md")
        assert result3.title == "fallback"

    def test_parse_sections(self, parser: MarkdownParser) -> None:
        """Test section parsing from headers."""
        content = """# Main Title

Intro paragraph.

## Section One

Content for section one.

## Section Two

Content for section two.

### Subsection

Nested content."""
        result = parser.parse_content(content, "/path/test.md")

        # Should have multiple sections
        assert len(result.sections) >= 3
        section_titles = [s.title for s in result.sections]
        assert "Main Title" in section_titles
        assert "Section One" in section_titles
        assert "Section Two" in section_titles

    def test_parse_empty_content(self, parser: MarkdownParser) -> None:
        """Test parsing empty content."""
        result = parser.parse_content("", "/path/empty.md")

        assert result.title == "empty"
        assert result.content == ""
        assert result.sections == []

    def test_parse_malformed_frontmatter(self, parser: MarkdownParser) -> None:
        """Test handling of malformed frontmatter."""
        content = """---
title: Good
invalid yaml: [unclosed
---

Content here."""
        # Should not raise, should handle gracefully
        result = parser.parse_content(content, "/path/test.md")
        assert result is not None


@pytest.mark.parametrize(
    "content, path, expected_title",
    [
        # Title from YAML front matter
        ("---\ntitle: YAML Title\n---\nRest of content", "/path/file.yaml.md", "YAML Title"),
        # Title from markdown header
        ("# Markdown Header Title\nMore content", "/path/file.md", "Markdown Header Title"),
        # Fallback to file stem when no title is found
        ("No title here at all", "/path/filename.md", "filename"),
    ],
)
def test_extract_title_variations(
    parser: MarkdownParser, content: str, path: str, expected_title: str
) -> None:
    """Test various title extraction scenarios."""
    result = parser.parse_content(content, path)
    assert result.title == expected_title

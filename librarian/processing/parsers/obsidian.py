"""
Obsidian-flavored markdown parser.

Extends the base markdown parser to handle Obsidian-specific syntax:
- Wiki-links: [[page]] and [[page|alias]]
- Tags: #tag and nested #parent/child
- Callouts: > [!type] content
- Embeds: ![[file]]
"""

import re
from typing import Any

from librarian.processing.parsers.md import MarkdownParser
from librarian.types import ParsedDocument


class ObsidianParser(MarkdownParser):
    """
    Parser for Obsidian markdown files.

    Extends MarkdownParser to extract Obsidian-specific elements
    and normalize them into the standard ParsedDocument format.
    """

    # Wiki-link patterns
    WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
    EMBED_PATTERN = re.compile(r"!\[\[([^\]]+)\]\]")

    # Tag pattern (including nested tags)
    TAG_PATTERN = re.compile(r"(?:^|\s)#([\w/]+)(?=\s|$|[^\w/])", re.MULTILINE)

    # Callout pattern
    CALLOUT_PATTERN = re.compile(r"^>\s*\[!(\w+)\][-+]?\s*(.*)$", re.MULTILINE)

    def parse_content(self, content: str, path: str = "") -> ParsedDocument:
        """
        Parse Obsidian markdown content.

        Extracts standard markdown plus Obsidian-specific elements
        (links, tags, callouts) into metadata.
        """
        # First, do standard markdown parsing
        doc = super().parse_content(content, path)

        # Extract Obsidian-specific elements
        wiki_links = self._extract_wiki_links(content)
        embeds = self._extract_embeds(content)
        tags = self._extract_tags(content)
        callouts = self._extract_callouts(content)

        # Merge into metadata
        obsidian_meta: dict[str, Any] = {}
        if wiki_links:
            obsidian_meta["wiki_links"] = wiki_links
        if embeds:
            obsidian_meta["embeds"] = embeds
        if tags:
            # Merge with frontmatter tags if present
            existing_tags = doc.metadata.get("tags", [])
            if isinstance(existing_tags, str):
                existing_tags = [existing_tags]
            obsidian_meta["tags"] = list(set(existing_tags + tags))
        if callouts:
            obsidian_meta["callouts"] = callouts

        # Update metadata
        if obsidian_meta:
            doc.metadata.update(obsidian_meta)

        return doc

    def _extract_wiki_links(self, content: str) -> list[dict[str, str]]:
        """
        Extract wiki-links from content.

        Handles both [[page]] and [[page|alias]] formats.

        Returns:
            List of dicts with 'target' and optional 'alias' keys.
        """
        links = []
        for match in self.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            alias = match.group(2)

            link_info: dict[str, str] = {"target": target}
            if alias:
                link_info["alias"] = alias.strip()
            links.append(link_info)

        return links

    def _extract_embeds(self, content: str) -> list[str]:
        """
        Extract embedded file references.

        Handles ![[file]] syntax for embedded images, notes, etc.

        Returns:
            List of embedded file paths/names.
        """
        return [match.group(1).strip() for match in self.EMBED_PATTERN.finditer(content)]

    def _extract_tags(self, content: str) -> list[str]:
        """
        Extract tags from content.

        Handles:
        - Simple tags: #tag
        - Nested tags: #parent/child
        - Tags anywhere in text (not just frontmatter)

        Returns:
            List of unique tag strings (without # prefix).
        """
        tags = set()
        for match in self.TAG_PATTERN.finditer(content):
            tag = match.group(1)
            tags.add(tag)
            # Also add parent tags for nested tags
            if "/" in tag:
                parts = tag.split("/")
                for i in range(1, len(parts)):
                    tags.add("/".join(parts[:i]))

        return sorted(tags)

    def _extract_callouts(self, content: str) -> list[dict[str, str]]:
        """
        Extract callout blocks.

        Handles Obsidian callout syntax:
        > [!type] optional title
        > content

        Returns:
            List of dicts with 'type' and 'title' keys.
        """
        callouts = []
        for match in self.CALLOUT_PATTERN.finditer(content):
            callout_type = match.group(1).lower()
            title = match.group(2).strip() if match.group(2) else ""
            callouts.append({"type": callout_type, "title": title})

        return callouts

    def normalize_wiki_links(self, content: str) -> str:
        """
        Convert wiki-links to standard markdown links.

        Useful for rendering or export. Converts:
        - [[page]] -> [page](page.md)
        - [[page|alias]] -> [alias](page.md)

        Args:
            content: Content with wiki-links.

        Returns:
            Content with standard markdown links.
        """

        def replace_link(match: re.Match[str]) -> str:
            target = match.group(1).strip()
            alias = match.group(2)
            display = alias.strip() if alias else target
            # Add .md extension if not present and no other extension
            if "." not in target:
                target = f"{target}.md"
            return f"[{display}]({target})"

        return self.WIKILINK_PATTERN.sub(replace_link, content)

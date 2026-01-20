"""Tests for text chunking functionality."""

import pytest

from librarian.processing.parsers.md import MarkdownParser
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy


@pytest.fixture
def chunker() -> Chunker:
    """Create a chunker instance with default settings."""
    return Chunker(chunk_size=200, chunk_overlap=20, min_chunk_size=20)


@pytest.fixture
def parser() -> MarkdownParser:
    """Create a markdown parser instance."""
    return MarkdownParser()


class TestChunker:
    """Tests for the Chunker class."""

    def test_chunk_short_text(self, chunker: Chunker) -> None:
        """Test chunking text shorter than chunk size."""
        text = "This is a short piece of text."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].index == 0

    def test_chunk_long_text(self, chunker: Chunker) -> None:
        """Test chunking text longer than chunk size."""
        # Create text that's definitely longer than chunk_size
        text = "This is a sentence. " * 50
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_chunk_positions(self, chunker: Chunker) -> None:
        """Test that chunk positions are tracked correctly."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)
            assert chunk.start_char < chunk.end_char

    def test_chunking_by_headers(self, chunker: Chunker, parser: MarkdownParser) -> None:
        """Test chunking by markdown headers."""
        content = """# Main Title

Introduction paragraph.

## Section One

Content for section one goes here.

## Section Two

Content for section two is different.
"""
        parsed = parser.parse_content(content, "/test.md")
        header_chunker = Chunker(
            chunk_size=500, chunk_overlap=50, strategy=ChunkingStrategy.HEADERS
        )
        chunks = header_chunker.chunk_document(parsed)

        assert len(chunks) >= 1
        # Each section should result in a chunk
        chunk_texts = [c.content for c in chunks]
        assert any("Introduction" in t for t in chunk_texts)

    def test_chunking_by_paragraphs(self) -> None:
        """Test chunking by paragraphs."""
        chunker = Chunker(chunk_size=200, chunk_overlap=20, strategy=ChunkingStrategy.PARAGRAPHS)
        text = """First paragraph with some content.

Second paragraph has different content.

Third paragraph completes the text."""

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()

    def test_chunking_by_sentences(self) -> None:
        """Test chunking by sentences."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20, strategy=ChunkingStrategy.SENTENCES)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1

    def test_fixed_chunking(self) -> None:
        """Test fixed-size chunking."""
        chunker = Chunker(chunk_size=50, chunk_overlap=10, strategy=ChunkingStrategy.FIXED)
        text = "A" * 200

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        # First chunk should be around chunk_size
        assert len(chunks[0].content) <= 50

    def test_merge_small_chunks(self) -> None:
        """Test that small chunks are merged."""
        # Create document with small sections
        text = "# A\n\nTiny.\n\n# B\n\nAlso tiny."

        parser = MarkdownParser()
        parsed = parser.parse_content(text, "/test.md")
        chunker_with_merge = Chunker(
            chunk_size=200,
            chunk_overlap=20,
            min_chunk_size=50,
            strategy=ChunkingStrategy.HEADERS,
        )
        chunks = chunker_with_merge.chunk_document(parsed)

        # Small chunks should be merged
        for chunk in chunks:
            # After merging, chunks should meet min size (unless it's the last one)
            if chunk != chunks[-1]:
                assert len(chunk.content) >= 20 or len(chunks) == 1

    def test_chunk_index_assignment(self, chunker: Chunker) -> None:
        """Test that chunk indices are properly assigned."""
        text = "Paragraph one. " * 20 + "\n\n" + "Paragraph two. " * 20
        chunks = chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_empty_text(self, chunker: Chunker) -> None:
        """Test chunking empty text."""
        chunks = chunker.chunk_text("")

        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0].content == "")

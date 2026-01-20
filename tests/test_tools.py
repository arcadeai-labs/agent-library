"""Tests for MCP tools."""

from pathlib import Path
from typing import Any

import pytest

# Context can be None in tests since we don't use it
CTX: Any = None


class TestIngestionTools:
    """Tests for document ingestion tools."""

    @pytest.mark.asyncio
    async def test_index_directory_to_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test indexing documents from a directory into the library."""
        from librarian.server import index_directory_to_library

        result = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )

        assert "error" not in result
        assert result["total_files"] >= 2
        assert result["indexed"] >= 0 or result["updated"] >= 0

    @pytest.mark.asyncio
    async def test_index_nonexistent_directory(self, clean_db: Path) -> None:
        """Test indexing from a nonexistent directory."""
        from librarian.server import index_directory_to_library

        result = await index_directory_to_library(
            context=CTX,
            directory="/nonexistent/path",
            recursive=True,
            force_reindex=False,
        )

        assert "error" in result
        assert result["indexed"] == 0

    @pytest.mark.asyncio
    async def test_index_change_detection(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test that unchanged files are skipped on re-index."""
        import time

        from librarian.server import index_directory_to_library

        # First index - should index all files
        result1 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result1
        initial_indexed = result1["indexed"]
        assert initial_indexed >= 2  # We have at least 2 test files

        # Second index without changes - should skip all
        result2 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result2
        assert result2["indexed"] == 0
        assert result2["updated"] == 0
        assert result2["skipped"] == result1["total_files"]

        # Modify one file (update mtime)
        time.sleep(0.1)  # Ensure mtime changes
        test_file = temp_docs_dir / "test1.md"
        original_content = test_file.read_text()
        test_file.write_text(original_content + "\n\nModified content.")

        # Third index - should only update the modified file
        result3 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result3
        assert result3["updated"] == 1
        assert result3["skipped"] == result1["total_files"] - 1

    @pytest.mark.asyncio
    async def test_index_force_reindex(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test that force_reindex re-indexes all files."""
        from librarian.server import index_directory_to_library

        # First index
        result1 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result1
        total_files = result1["total_files"]

        # Force reindex - should update all even though nothing changed
        result2 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )
        assert "error" not in result2
        assert result2["updated"] == total_files
        assert result2["skipped"] == 0

    @pytest.mark.asyncio
    async def test_add_to_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding new content to the library."""
        from librarian.server import add_to_library

        result = await add_to_library(
            context=CTX,
            content="# New Document\n\nThis is new content.",
            title="new_doc",
            directory=str(temp_docs_dir),
        )

        assert result.get("status") == "stored"
        assert "new_doc.md" in result.get("path", "")
        assert (temp_docs_dir / "new_doc.md").exists()

    @pytest.mark.asyncio
    async def test_add_to_library_with_tags(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding content to the library with tags."""
        from librarian.server import add_to_library

        result = await add_to_library(
            context=CTX,
            content="Content here.",
            title="with_tags",
            directory=str(temp_docs_dir),
            tags=["test", "example"],
            metadata={"author": "tester"},
        )

        assert result.get("status") == "stored"
        content = (temp_docs_dir / "with_tags.md").read_text()
        assert "---" in content
        assert "tags:" in content

    @pytest.mark.asyncio
    async def test_add_to_library_already_exists(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding content that already exists."""
        from librarian.server import add_to_library

        result = await add_to_library(
            context=CTX,
            content="New content.",
            title="test1",  # Already exists
            directory=str(temp_docs_dir),
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_library_doc(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test updating existing content in the library."""
        from librarian.server import update_library_doc

        file_path = temp_docs_dir / "test1.md"
        result = await update_library_doc(
            context=CTX,
            path=str(file_path),
            content="# Updated Title\n\nUpdated content.",
        )

        assert result.get("status") == "updated"
        new_content = file_path.read_text()
        assert "Updated" in new_content


class TestSearchTools:
    """Tests for search tools."""

    @pytest.mark.asyncio
    async def test_search_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test hybrid search in the library."""
        from librarian.server import index_directory_to_library, search_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        results = await search_library(context=CTX, query="test document", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_semantic_search_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test pure semantic search in the library."""
        from librarian.server import index_directory_to_library, semantic_search_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        results = await semantic_search_library(context=CTX, query="document content", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_keyword_search_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test keyword search in the library."""
        from librarian.server import index_directory_to_library, keyword_search_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        results = await keyword_search_library(context=CTX, query="test", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_library_empty_query(self, clean_db: Path) -> None:
        """Test search with empty query."""
        from librarian.server import search_library

        results = await search_library(context=CTX, query="", limit=5)
        assert results == []


class TestDocumentManagementTools:
    """Tests for document management tools."""

    @pytest.mark.asyncio
    async def test_read_from_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test reading content from the library."""
        from librarian.server import index_directory_to_library, read_from_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        file_path = temp_docs_dir / "test1.md"
        result = await read_from_library(context=CTX, path=str(file_path))

        assert "error" not in result
        assert "content" in result
        assert "Test Document 1" in result["content"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_from_library(self, clean_db: Path) -> None:
        """Test reading nonexistent content from the library."""
        from librarian.server import read_from_library

        result = await read_from_library(context=CTX, path="/nonexistent/file.md")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_remove_from_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test removing content from the library."""
        from librarian.server import index_directory_to_library, remove_from_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        file_path = temp_docs_dir / "test1.md"
        result = await remove_from_library(context=CTX, path=str(file_path), delete_file=True)

        assert result.get("file_deleted") is True
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_list_library_contents(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test listing all library contents."""
        from librarian.server import index_directory_to_library, list_library_contents

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        result = await list_library_contents(context=CTX, limit=100)
        assert isinstance(result, list)
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_get_library_stats(self, clean_db: Path) -> None:
        """Test getting library statistics."""
        from librarian.server import get_library_stats

        result = await get_library_stats(context=CTX)
        assert "document_count" in result
        assert "chunk_count" in result
        assert "config" in result

    @pytest.mark.asyncio
    async def test_get_library_sources(self, clean_db: Path) -> None:
        """Test getting library sources."""
        from librarian.server import get_library_sources

        result = await get_library_sources(context=CTX)
        # Result should be a list (may be empty if no sources registered)
        assert isinstance(result, list)

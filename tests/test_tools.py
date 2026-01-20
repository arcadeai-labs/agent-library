"""Tests for MCP tools."""

from pathlib import Path
from typing import Any

import pytest

# Context can be None in tests since we don't use it
CTX: Any = None


class TestIngestionTools:
    """Tests for document ingestion tools."""

    @pytest.mark.asyncio
    async def test_ingest_directory(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test ingesting documents from a directory."""
        from librarian.server import ingest_directory

        result = await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )

        assert "error" not in result
        assert result["total_files"] >= 2
        assert result["indexed"] >= 0 or result["updated"] >= 0

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_directory(self, clean_db: Path) -> None:
        """Test ingesting from a nonexistent directory."""
        from librarian.server import ingest_directory

        result = await ingest_directory(
            context=CTX,
            directory="/nonexistent/path",
            recursive=True,
            force_reindex=False,
        )

        assert "error" in result
        assert result["indexed"] == 0

    @pytest.mark.asyncio
    async def test_ingest_change_detection(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test that unchanged files are skipped on re-ingest."""
        import time

        from librarian.server import ingest_directory

        # First ingest - should index all files
        result1 = await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result1
        initial_indexed = result1["indexed"]
        assert initial_indexed >= 2  # We have at least 2 test files

        # Second ingest without changes - should skip all
        result2 = await ingest_directory(
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

        # Third ingest - should only update the modified file
        result3 = await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result3
        assert result3["updated"] == 1
        assert result3["skipped"] == result1["total_files"] - 1

    @pytest.mark.asyncio
    async def test_ingest_force_reindex(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test that force_reindex re-indexes all files."""
        from librarian.server import ingest_directory

        # First ingest
        result1 = await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=False,
        )
        assert "error" not in result1
        total_files = result1["total_files"]

        # Force reindex - should update all even though nothing changed
        result2 = await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )
        assert "error" not in result2
        assert result2["updated"] == total_files
        assert result2["skipped"] == 0

    @pytest.mark.asyncio
    async def test_add_document(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding a new document."""
        from librarian.server import add_document

        result = await add_document(
            context=CTX,
            content="# New Document\n\nThis is new content.",
            filename="new_doc.md",
            directory=str(temp_docs_dir),
        )

        assert result.get("status") == "success"
        assert "new_doc.md" in result.get("path", "")
        assert (temp_docs_dir / "new_doc.md").exists()

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding a document with metadata."""
        from librarian.server import add_document

        result = await add_document(
            context=CTX,
            content="Content here.",
            filename="with_meta.md",
            directory=str(temp_docs_dir),
            metadata={"tags": ["test"], "author": "tester"},
        )

        assert result.get("status") == "success"
        content = (temp_docs_dir / "with_meta.md").read_text()
        assert "---" in content
        assert "tags:" in content

    @pytest.mark.asyncio
    async def test_add_document_already_exists(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding a document that already exists."""
        from librarian.server import add_document

        result = await add_document(
            context=CTX,
            content="New content.",
            filename="test1.md",  # Already exists
            directory=str(temp_docs_dir),
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_document(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test updating an existing document."""
        from librarian.server import update_document

        file_path = temp_docs_dir / "test1.md"
        result = await update_document(
            context=CTX,
            path=str(file_path),
            content="# Updated Title\n\nUpdated content.",
        )

        assert result.get("status") == "success"
        new_content = file_path.read_text()
        assert "Updated" in new_content


class TestSearchTools:
    """Tests for search tools."""

    @pytest.mark.asyncio
    async def test_search(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test hybrid search."""
        from librarian.server import ingest_directory, search

        await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        results = await search(context=CTX, query="test document", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_vector_search(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test pure vector search."""
        from librarian.server import ingest_directory, vector_search

        await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        results = await vector_search(context=CTX, query="document content", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_keyword_search(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test keyword search."""
        from librarian.server import ingest_directory, keyword_search

        await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        results = await keyword_search(context=CTX, query="test", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_empty_query(self, clean_db: Path) -> None:
        """Test search with empty query."""
        from librarian.server import search

        results = await search(context=CTX, query="", limit=5)
        assert results == []


class TestDocumentManagementTools:
    """Tests for document management tools."""

    @pytest.mark.asyncio
    async def test_read_document(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test reading a document."""
        from librarian.server import ingest_directory, read_document

        await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        file_path = temp_docs_dir / "test1.md"
        result = await read_document(context=CTX, path=str(file_path))

        assert "error" not in result
        assert "content" in result
        assert "Test Document 1" in result["content"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_document(self, clean_db: Path) -> None:
        """Test reading a nonexistent document."""
        from librarian.server import read_document

        result = await read_document(context=CTX, path="/nonexistent/file.md")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_delete_document(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test deleting a document."""
        from librarian.server import delete_document, ingest_directory

        await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        file_path = temp_docs_dir / "test1.md"
        result = await delete_document(context=CTX, path=str(file_path), delete_file=True)

        assert result.get("file_deleted") is True
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_list_documents(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test listing documents."""
        from librarian.server import ingest_directory, list_documents

        await ingest_directory(
            context=CTX,
            directory=str(temp_docs_dir),
            recursive=True,
            force_reindex=True,
        )

        result = await list_documents(context=CTX, limit=100)
        assert isinstance(result, list)
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_get_stats(self, clean_db: Path) -> None:
        """Test getting index statistics."""
        from librarian.server import get_stats

        result = await get_stats(context=CTX)
        assert "document_count" in result
        assert "chunk_count" in result
        assert "config" in result

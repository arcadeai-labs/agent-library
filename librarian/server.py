#!/usr/bin/env python3
"""
Librarian MCP Server - Agent Knowledge Library.

A personal knowledge library for AI agents to store, search, and retrieve
any text, documents, notes, and information. Think of it as an agent's
personal library that persists across sessions.

Usage:
    uv run librarian/server.py stdio    # For Claude Desktop, CLI tools
    uv run librarian/server.py http     # For Cursor, VS Code (HTTP streaming)
"""

import logging
import sys
from pathlib import Path
from typing import Annotated, Any

from arcade_mcp_server import Context, MCPApp

from librarian.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_PATH,
    HYBRID_ALPHA,
    MMR_LAMBDA,
    SEARCH_LIMIT,
    ensure_directories,
)
from librarian.indexing import get_indexing_service
from librarian.processing.embed import get_embedder
from librarian.retrieval.search import HybridSearcher
from librarian.storage.database import get_database
from librarian.utils.timeframe import Timeframe, get_timeframe_bounds, parse_date_string

logger = logging.getLogger(__name__)

# Create the MCP application
app = MCPApp(
    name="Librarian",
    version="0.5.0",
    log_level="INFO",
)

# Ensure required directories exist
ensure_directories()


# =============================================================================
# Helper Functions
# =============================================================================


def _get_searcher() -> HybridSearcher:
    """Get a configured hybrid searcher instance."""
    embedder = get_embedder()
    return HybridSearcher(embedder)


def _process_and_index_file(file_path: Path) -> dict[str, Any]:
    """Process a markdown file and add it to the index."""
    return get_indexing_service().index_file(file_path)


# =============================================================================
# Library Ingestion Tools
# =============================================================================


@app.tool
async def index_directory_to_library(
    context: Context,
    directory: Annotated[str, "Path to directory containing files to add to the library"] = "",
    recursive: Annotated[bool, "Whether to include subdirectories"] = True,
    force_reindex: Annotated[bool, "Whether to re-process already indexed documents"] = False,
) -> dict[str, Any]:
    """
    Add all documents from a directory to the agent's library.

    Use this to bulk-import text files, notes, documentation, or any
    markdown content into the library for later search and retrieval.
    The library persists across sessions, so content added here will
    be available in future conversations.
    """
    dir_path = Path(directory) if directory else Path(DOCUMENTS_PATH)

    if not dir_path.exists():
        return {"error": f"Directory not found: {dir_path}", "indexed": 0}

    if not dir_path.is_dir():
        return {"error": f"Not a directory: {dir_path}", "indexed": 0}

    # Find markdown files
    pattern = "**/*.md" if recursive else "*.md"
    md_files = list(dir_path.glob(pattern))

    results: dict[str, Any] = {
        "directory": str(dir_path),
        "total_files": len(md_files),
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
        "files": [],
    }

    db = get_database()

    for file_path in md_files:
        try:
            # Check if document exists and compare modification times
            existing = db.get_document_by_path(str(file_path))

            if existing and not force_reindex:
                # Get current file modification time
                current_mtime = file_path.stat().st_mtime
                stored_mtime = existing.file_mtime

                # Skip if file hasn't been modified since last index
                if stored_mtime is not None and current_mtime <= stored_mtime:
                    results["files"].append({
                        "path": str(file_path),
                        "status": "skipped",
                    })
                    results["skipped"] += 1
                    continue

            result = _process_and_index_file(file_path)
            results["files"].append(result)

            if result["status"] == "created":
                results["indexed"] += 1
            else:
                results["updated"] += 1

        except Exception as e:
            logger.exception("Error indexing %s", file_path)
            results["errors"].append({"path": str(file_path), "error": str(e)})

    return results


@app.tool
async def add_to_library(
    context: Context,
    content: Annotated[str, "The text content to store in the library"],
    title: Annotated[str, "A title or filename for this content (without .md extension)"],
    directory: Annotated[str, "Directory to save in (optional, uses default library path)"] = "",
    tags: Annotated[list[str] | None, "Optional tags to categorize this content"] = None,
    metadata: Annotated[dict[str, Any] | None, "Optional additional metadata"] = None,
) -> dict[str, Any]:
    """
    Store new content in the agent's library.

    Use this to save any text, notes, information, or knowledge that should
    be remembered and searchable later. The content is indexed for both
    semantic (meaning-based) and keyword search, making it easy to find
    relevant information in future conversations.

    Examples of what to store:
    - Meeting notes and summaries
    - Research findings and insights
    - Code documentation and explanations
    - Personal notes and reminders
    - Any information worth remembering
    """
    dir_path = Path(directory) if directory else Path(DOCUMENTS_PATH)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Clean filename and ensure .md extension
    filename = title.replace("/", "-").replace("\\", "-")
    if not filename.endswith(".md"):
        filename = filename + ".md"

    file_path = dir_path / filename

    # Check if file already exists
    if file_path.exists():
        return {
            "error": f"Content with this title already exists: {title}",
            "path": str(file_path),
            "suggestion": "Use update_library_doc to modify existing content",
        }

    # Build metadata
    meta = metadata or {}
    if tags:
        meta["tags"] = tags

    # Add frontmatter if metadata provided
    if meta:
        import yaml

        frontmatter_str = "---\n" + yaml.dump(meta, default_flow_style=False) + "---\n\n"
        content = frontmatter_str + content

    # Write the file
    file_path.write_text(content, encoding="utf-8")

    # Index it
    try:
        result = _process_and_index_file(file_path)
        return {
            "status": "stored",
            "message": f"Content '{title}' has been added to your library",
            "path": str(file_path),
            "title": result["title"],
            "chunks": result["chunks"],
        }
    except Exception as e:
        # Clean up file on error
        file_path.unlink(missing_ok=True)
        return {"error": str(e), "path": str(file_path)}


@app.tool
async def update_library_doc(
    context: Context,
    path: Annotated[str, "Path to the document to update"],
    content: Annotated[str, "The new content to replace the existing content"],
) -> dict[str, Any]:
    """
    Update existing content in the agent's library.

    Use this to modify or replace content that was previously stored.
    The updated content will be re-indexed for search.
    """
    file_path = Path(path)

    if not file_path.exists():
        return {"error": f"Document not found in library: {path}"}

    # Write new content
    file_path.write_text(content, encoding="utf-8")

    # Reindex
    try:
        result = _process_and_index_file(file_path)
        return {
            "status": "updated",
            "message": "Content has been updated in your library",
            "path": str(file_path),
            "title": result["title"],
            "chunks": result["chunks"],
        }
    except Exception as e:
        return {"error": str(e), "path": str(file_path)}


# =============================================================================
# Library Search Tools
# =============================================================================


@app.tool
async def search_library(
    context: Context,
    query: Annotated[str, "What to search for in the library"],
    limit: Annotated[int, "Maximum number of results to return"] = 10,
    use_mmr: Annotated[bool, "Use diverse results (recommended)"] = True,
    hybrid_alpha: Annotated[
        float, "Balance between meaning (1.0) and keywords (0.0)"
    ] = 0.7,
    timeframe: Annotated[
        str | None,
        "Filter by time: today, yesterday, this_week, last_week, "
        "this_month, last_month, last_7_days, last_30_days, this_year",
    ] = None,
) -> list[dict[str, Any]]:
    """
    Search the agent's library for relevant information.

    This is the primary way to find stored knowledge. It uses both
    semantic understanding (finding content with similar meaning) and
    keyword matching to find the most relevant results.

    Use this when you need to:
    - Find previously stored information
    - Look up notes or documentation
    - Recall context from past conversations
    - Answer questions using stored knowledge
    """
    if not query.strip():
        return []

    db = get_database()
    filter_doc_ids: list[int] | None = None

    # Apply timeframe filter if specified
    if timeframe:
        try:
            tf = Timeframe(timeframe)
            start_date, end_date = get_timeframe_bounds(tf)
            filter_doc_ids = db.get_document_ids_in_timerange(start_date, end_date)
            if not filter_doc_ids:
                return []  # No documents in timeframe
        except ValueError:
            pass  # Invalid timeframe, ignore filter

    searcher = _get_searcher()
    searcher.hybrid_alpha = hybrid_alpha

    results = searcher.search(
        query, limit=limit, use_mmr=use_mmr, filter_document_ids=filter_doc_ids
    )

    return [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "document_path": r.document_path,
            "content": r.content,
            "heading_path": r.heading_path,
            "score": round(r.score, 4),
            "snippet": r.snippet,
        }
        for r in results
    ]


@app.tool
async def search_library_by_dates(
    context: Context,
    query: Annotated[str, "What to search for"],
    start_date: Annotated[str, "Start date (YYYY-MM-DD)"],
    end_date: Annotated[str, "End date (YYYY-MM-DD)"],
    limit: Annotated[int, "Maximum number of results"] = 10,
    use_mmr: Annotated[bool, "Use diverse results"] = True,
) -> list[dict[str, Any]]:
    """
    Search the library for content within a specific date range.

    Use this when you need to find information that was stored or
    updated during a particular time period.
    """
    if not query.strip():
        return []

    # Parse dates
    start_dt = parse_date_string(start_date)
    end_dt = parse_date_string(end_date)

    if not start_dt:
        return [{"error": f"Invalid start_date format: {start_date}"}]
    if not end_dt:
        return [{"error": f"Invalid end_date format: {end_date}"}]

    # If end_date has no time component, set to end of day
    if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    db = get_database()
    filter_doc_ids = db.get_document_ids_in_timerange(start_dt, end_dt)

    if not filter_doc_ids:
        return []

    searcher = _get_searcher()
    results = searcher.search(
        query, limit=limit, use_mmr=use_mmr, filter_document_ids=filter_doc_ids
    )

    return [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "document_path": r.document_path,
            "content": r.content,
            "heading_path": r.heading_path,
            "score": round(r.score, 4),
            "snippet": r.snippet,
            "date_range": {"start": start_date, "end": end_date},
        }
        for r in results
    ]


@app.tool
async def semantic_search_library(
    context: Context,
    query: Annotated[str, "What to search for (searches by meaning)"],
    limit: Annotated[int, "Maximum number of results"] = 10,
) -> list[dict[str, Any]]:
    """
    Search the library using semantic similarity only.

    This finds content that has similar meaning to your query,
    even if it doesn't contain the exact words. Best for:
    - Finding related concepts
    - Discovering content you might not find with keywords
    - Understanding and inference-based search
    """
    if not query.strip():
        return []

    searcher = _get_searcher()
    results = searcher.vector_search(query, limit=limit)

    return [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "document_path": r.document_path,
            "content": r.content,
            "heading_path": r.heading_path,
            "score": round(r.score, 4),
        }
        for r in results
    ]


@app.tool
async def keyword_search_library(
    context: Context,
    query: Annotated[str, "Keywords to search for"],
    limit: Annotated[int, "Maximum number of results"] = 10,
) -> list[dict[str, Any]]:
    """
    Search the library using exact keyword matching.

    This finds content containing the specific words in your query.
    Best for:
    - Finding specific terms or phrases
    - Technical searches where exact wording matters
    - When you know exactly what you're looking for
    """
    if not query.strip():
        return []

    searcher = _get_searcher()
    results = searcher.keyword_search(query, limit=limit)

    return [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "document_path": r.document_path,
            "content": r.content,
            "heading_path": r.heading_path,
            "score": round(r.score, 4),
            "snippet": r.snippet,
        }
        for r in results
    ]


# =============================================================================
# Library Management Tools
# =============================================================================


@app.tool
async def read_from_library(
    context: Context,
    path: Annotated[str, "Path to the document to read"],
) -> dict[str, Any]:
    """
    Read the full content of a document from the library.

    Use this after searching to get the complete content of a
    document, rather than just the matching snippets.
    """
    db = get_database()
    doc = db.get_document_by_path(path)

    if not doc:
        # Try reading from filesystem
        file_path = Path(path)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return {
                "path": path,
                "content": content,
                "indexed": False,
                "note": "This file exists but is not in the library index",
            }
        return {"error": f"Document not found: {path}"}

    return {
        "id": doc.id,
        "path": doc.path,
        "title": doc.title,
        "content": doc.content,
        "metadata": doc.metadata,
        "created_at": str(doc.created_at) if doc.created_at else None,
        "updated_at": str(doc.updated_at) if doc.updated_at else None,
        "indexed": True,
    }


@app.tool
async def remove_from_library(
    context: Context,
    path: Annotated[str, "Path to the document to remove"],
    delete_file: Annotated[bool, "Also delete the file from disk (permanent)"] = False,
) -> dict[str, Any]:
    """
    Remove a document from the agent's library.

    By default, this only removes from the search index (the file
    remains on disk). Set delete_file=True to permanently delete.
    """
    db = get_database()

    # Remove from index
    deleted = db.delete_document_by_path(path)

    result = {
        "path": path,
        "removed_from_index": deleted,
        "message": "Document removed from library" if deleted else "Document was not in library",
    }

    # Optionally delete file
    if delete_file:
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
            result["file_deleted"] = True
            result["message"] = "Document permanently deleted"
        else:
            result["file_deleted"] = False

    return result


@app.tool
async def list_library_contents(
    context: Context,
    limit: Annotated[int, "Maximum number of documents to list"] = 100,
) -> list[dict[str, Any]]:
    """
    List all documents stored in the agent's library.

    Returns a summary of each document including title, path,
    and when it was added/updated.
    """
    db = get_database()
    documents = db.list_documents()[:limit]

    return [
        {
            "id": doc.id,
            "path": doc.path,
            "title": doc.title,
            "metadata": doc.metadata,
            "created_at": str(doc.created_at) if doc.created_at else None,
            "updated_at": str(doc.updated_at) if doc.updated_at else None,
        }
        for doc in documents
    ]


@app.tool
async def get_library_stats(context: Context) -> dict[str, Any]:
    """
    Get statistics about the agent's library.

    Shows how many documents are stored, total chunks indexed,
    and current configuration settings.
    """
    db = get_database()
    stats = db.get_stats()

    return {
        **stats,
        "config": {
            "documents_path": DOCUMENTS_PATH,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "search_limit": SEARCH_LIMIT,
            "mmr_lambda": MMR_LAMBDA,
            "hybrid_alpha": HYBRID_ALPHA,
        },
    }


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    transport_arg = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport_arg not in ("http", "stdio"):
        transport_arg = "stdio"
    app.run(transport=transport_arg, host="127.0.0.1", port=8000)  # type: ignore[arg-type]

#!/usr/bin/env python3
"""
Librarian MCP Server.

A markdown document management system for maintaining, indexing,
ingesting, and retrieving markdown documents through LLM tools.

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
# Ingestion Tools
# =============================================================================


@app.tool
async def ingest_directory(
    context: Context,
    directory: Annotated[str, "Path to directory containing markdown files"] = "",
    recursive: Annotated[bool, "Whether to search subdirectories"] = True,
    force_reindex: Annotated[bool, "Whether to reindex existing documents"] = False,
) -> dict[str, Any]:
    """
    Ingest markdown files from a directory into the index.

    Scans the specified directory for markdown files, parses them,
    generates embeddings, and stores them in the vector database.
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
async def add_document(
    context: Context,
    content: Annotated[str, "The markdown content to add"],
    filename: Annotated[str, "Filename for the document (without path)"],
    directory: Annotated[str, "Directory to save the document in"] = "",
    metadata: Annotated[dict[str, Any] | None, "Optional metadata for the document"] = None,
) -> dict[str, Any]:
    """
    Add a new markdown document to the index.

    Creates a new markdown file with the given content and indexes it
    for searching.
    """
    dir_path = Path(directory) if directory else Path(DOCUMENTS_PATH)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure .md extension
    if not filename.endswith(".md"):
        filename = filename + ".md"

    file_path = dir_path / filename

    # Check if file already exists
    if file_path.exists():
        return {
            "error": f"File already exists: {file_path}",
            "path": str(file_path),
        }

    # Add frontmatter if metadata provided
    if metadata:
        import yaml

        frontmatter_str = "---\n" + yaml.dump(metadata, default_flow_style=False) + "---\n\n"
        content = frontmatter_str + content

    # Write the file
    file_path.write_text(content, encoding="utf-8")

    # Index it
    try:
        result = _process_and_index_file(file_path)
        return {
            "status": "success",
            "path": str(file_path),
            "title": result["title"],
            "chunks": result["chunks"],
        }
    except Exception as e:
        # Clean up file on error
        file_path.unlink(missing_ok=True)
        return {"error": str(e), "path": str(file_path)}


@app.tool
async def update_document(
    context: Context,
    path: Annotated[str, "Path to the document to update"],
    content: Annotated[str, "The new markdown content"],
) -> dict[str, Any]:
    """
    Update an existing document's content and reindex it.
    """
    file_path = Path(path)

    if not file_path.exists():
        return {"error": f"File not found: {path}"}

    # Write new content
    file_path.write_text(content, encoding="utf-8")

    # Reindex
    try:
        result = _process_and_index_file(file_path)
        return {
            "status": "success",
            "path": str(file_path),
            "title": result["title"],
            "chunks": result["chunks"],
        }
    except Exception as e:
        return {"error": str(e), "path": str(file_path)}


# =============================================================================
# Search Tools
# =============================================================================


@app.tool
async def search(
    context: Context,
    query: Annotated[str, "The search query"],
    limit: Annotated[int, "Maximum number of results to return"] = 10,
    use_mmr: Annotated[bool, "Use Max Marginal Relevance for diverse results"] = True,
    hybrid_alpha: Annotated[
        float, "Weight for vector vs keyword search (0=keyword, 1=vector)"
    ] = 0.7,
    timeframe: Annotated[
        str | None,
        "Optional timeframe filter: today, yesterday, this_week, last_week, "
        "this_month, last_month, last_7_days, last_30_days, last_90_days, "
        "this_year, last_year",
    ] = None,
) -> list[dict[str, Any]]:
    """
    Search for relevant document chunks using hybrid vector and keyword search.

    Combines semantic similarity search with full-text keyword search,
    optionally using MMR for diverse results. Can be filtered by timeframe.
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
async def find_relevant_context_within_specific_dates(
    context: Context,
    query: Annotated[str, "The search query"],
    start_date: Annotated[str, "Start date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"],
    end_date: Annotated[str, "End date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"],
    limit: Annotated[int, "Maximum number of results to return"] = 10,
    use_mmr: Annotated[bool, "Use Max Marginal Relevance for diverse results"] = True,
) -> list[dict[str, Any]]:
    """
    Search for relevant context within a specific date range.

    Use this tool when you need to find documents that were created or updated
    between specific dates. Dates should be in ISO format.

    Example:
        start_date: "2025-01-01"
        end_date: "2025-01-15"
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
async def vector_search(
    context: Context,
    query: Annotated[str, "The search query"],
    limit: Annotated[int, "Maximum number of results"] = 10,
) -> list[dict[str, Any]]:
    """
    Search using pure vector similarity (semantic search).

    Uses embeddings to find semantically similar content.
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
async def keyword_search(
    context: Context,
    query: Annotated[str, "The keyword search query"],
    limit: Annotated[int, "Maximum number of results"] = 10,
) -> list[dict[str, Any]]:
    """
    Search using pure keyword/full-text search.

    Uses SQLite FTS5 for fast keyword matching with BM25 ranking.
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
# Document Management Tools
# =============================================================================


@app.tool
async def read_document(
    context: Context,
    path: Annotated[str, "Path to the document to read"],
) -> dict[str, Any]:
    """
    Read the full content of a document.
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
async def delete_document(
    context: Context,
    path: Annotated[str, "Path to the document to delete"],
    delete_file: Annotated[bool, "Also delete the file from disk"] = False,
) -> dict[str, Any]:
    """
    Delete a document from the index.

    Optionally also deletes the file from disk.
    """
    db = get_database()

    # Remove from index
    deleted = db.delete_document_by_path(path)

    result = {
        "path": path,
        "removed_from_index": deleted,
    }

    # Optionally delete file
    if delete_file:
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
            result["file_deleted"] = True
        else:
            result["file_deleted"] = False

    return result


@app.tool
async def list_documents(
    context: Context,
    limit: Annotated[int, "Maximum number of documents to list"] = 100,
) -> list[dict[str, Any]]:
    """
    List all indexed documents.
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
async def get_stats(context: Context) -> dict[str, Any]:
    """
    Get statistics about the index.

    Returns counts of documents, chunks, and embeddings,
    plus configuration information.
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

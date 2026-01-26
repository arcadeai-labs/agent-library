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


def _should_skip_file(file_path: Path, supported_extensions: set[str]) -> bool:
    """
    Check if a file should be skipped during indexing.

    Args:
        file_path: Path to the file.
        supported_extensions: Set of supported extensions.

    Returns:
        True if the file should be skipped.
    """
    # Skip system/hidden directories
    skip_dirs = {
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
    }

    # Check if file is in a skipped directory
    for parent in file_path.parents:
        if parent.name in skip_dirs:
            return True

    # Skip hidden files (starting with .)
    if file_path.name.startswith("."):
        return True

    # Skip binary/system file extensions
    skip_extensions = {
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
    }

    if file_path.suffix.lower() in skip_extensions:
        return True

    # Skip files without extensions
    if not file_path.suffix:
        return True

    # Skip if extension not in supported list
    return file_path.suffix.lower() not in supported_extensions


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

    # Find all supported files
    from librarian.processing.parsers.registry import get_registry

    registry = get_registry()
    supported_extensions = registry.get_supported_extensions()

    all_files: list[Path] = []
    for ext in supported_extensions:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        all_files.extend(dir_path.glob(pattern))

    # Filter out system/binary files
    all_files = [f for f in all_files if not _should_skip_file(f, supported_extensions)]

    results: dict[str, Any] = {
        "directory": str(dir_path),
        "total_files": len(all_files),
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
        "files": [],
    }

    db = get_database()

    for file_path in all_files:
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


def _get_sources_config() -> list[dict[str, Any]]:
    """
    Load sources configuration from the sources.json file.

    Returns:
        List of source configuration dictionaries, or empty list if file
        doesn't exist or is invalid.
    """
    import json

    sources_file = Path.home() / ".librarian" / "sources.json"
    if not sources_file.exists():
        return []
    try:
        with open(sources_file) as f:
            result = json.load(f)
            return result if isinstance(result, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _find_source_for_path(path: Path, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find which source a path belongs to."""
    path_str = str(path.resolve())
    for source in sources:
        source_path = source.get("path", "")
        if path_str.startswith(source_path):
            return source
    return None


def _build_breadcrumb(path: Path, source: dict[str, Any] | None) -> list[str]:
    """Build breadcrumb path from source root to file."""
    if not source:
        return [path.name]

    source_path = Path(source.get("path", ""))
    try:
        relative = path.relative_to(source_path)
    except ValueError:
        return [path.name]
    else:
        return [source.get("name", source_path.name), *relative.parts]


def _get_siblings(path: Path, include_dirs: bool = True) -> list[dict[str, str]]:
    """
    Get sibling files and directories in the same parent folder.

    Args:
        path: The file path to find siblings for.
        include_dirs: Whether to include directories in output.

    Returns:
        List of sibling items (max 20) with name and type.
    """
    parent = path.parent
    if not parent.exists():
        return []

    siblings = []
    try:
        for item in sorted(parent.iterdir()):
            if item.name.startswith("."):
                continue
            if item == path:
                continue
            if item.is_dir() and include_dirs:
                siblings.append({"name": item.name, "type": "directory"})
            elif item.is_file() and item.suffix == ".md":
                siblings.append({"name": item.name, "type": "file"})
    except PermissionError:
        return []

    return siblings[:20]


@app.tool
async def add_to_library(
    context: Context,
    content: Annotated[str, "The text content to store in the library"],
    title: Annotated[str, "A title or filename for this content (without .md extension)"],
    directory: Annotated[
        str, "Full path to directory (use get_library_sections to find valid paths)"
    ] = "",
    tags: Annotated[list[str] | None, "Optional tags to categorize this content"] = None,
    metadata: Annotated[dict[str, Any] | None, "Optional additional metadata"] = None,
) -> dict[str, Any]:
    """
    Store new content in the agent's library.

    IMPORTANT: Before adding content, use get_library_sections() to see available
    locations and their purposes. Pass the full directory path from that tool.

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

    # Build location context (always available even if indexing fails)
    sources = _get_sources_config()
    source = _find_source_for_path(file_path, sources)
    breadcrumb = _build_breadcrumb(file_path, source)
    siblings = _get_siblings(file_path)

    location_info = {
        "source": source.get("name") if source else None,
        "source_path": source.get("path") if source else None,
        "directory": str(dir_path),
        "breadcrumb": breadcrumb,
        "breadcrumb_display": " → ".join(breadcrumb),
    }

    # Try to index with embeddings
    try:
        result = _process_and_index_file(file_path)
        return {
            "status": "stored",
            "message": f"Content '{title}' has been added to your library",
            "path": str(file_path),
            "title": result["title"],
            "chunks": result["chunks"],
            "indexed": True,
            "location": location_info,
            "context": {
                "siblings": siblings,
                "sibling_count": len(siblings),
            },
        }
    except Exception as e:
        # Indexing failed (likely embedding service unavailable)
        # File is still saved - try to store document metadata without embeddings
        logger.warning("Full indexing failed, storing without embeddings: %s", e)

        try:
            # Store document in database without embeddings for FTS search
            db = get_database()
            from librarian.processing.parsers.md import MarkdownParser
            from librarian.types import Document

            parser = MarkdownParser()
            parsed = parser.parse_file(file_path)
            file_mtime = file_path.stat().st_mtime

            doc = Document(
                id=None,
                path=str(file_path),
                title=parsed.title,
                content=parsed.content,
                metadata=parsed.metadata,
                file_mtime=file_mtime,
            )
            db.insert_document(doc)

            return {
                "status": "stored_partial",
                "message": f"Content '{title}' saved but indexing incomplete (embedding service unavailable)",
                "path": str(file_path),
                "title": parsed.title,
                "chunks": 0,
                "indexed": False,
                "warning": "File saved but not fully indexed. Keyword search will work, but semantic search won't find this content until re-indexed.",
                "location": location_info,
                "context": {
                    "siblings": siblings,
                    "sibling_count": len(siblings),
                },
            }
        except Exception:
            # Even basic storage failed - but file is still on disk
            logger.exception("Failed to store document metadata")
            return {
                "status": "stored_file_only",
                "message": f"Content '{title}' saved to disk but not indexed",
                "path": str(file_path),
                "indexed": False,
                "error": str(e),
                "location": location_info,
            }


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
    hybrid_alpha: Annotated[float, "Balance between meaning (1.0) and keywords (0.0)"] = 0.7,
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
            "asset_type": r.asset_type.value,
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
            "asset_type": r.asset_type.value,
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
            "asset_type": r.asset_type.value,
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
            "asset_type": r.asset_type.value,
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
async def get_library_sources(context: Context) -> list[dict[str, Any]]:
    """
    List all sources in the agent's library with statistics.

    Shows each registered source (directory or file) along with:
    - Number of documents indexed from that source
    - Number of chunks generated
    - Source path and name
    - Whether the source is still accessible

    Use this to understand what knowledge is in the library and where it came from.
    """
    sources = _get_sources_config()
    if not sources:
        return []

    db = get_database()
    all_documents = db.list_documents()

    result = []
    for source in sources:
        source_path = source.get("path", "")
        source_name = source.get("name", source_path)

        # Count documents from this source
        docs_from_source = [d for d in all_documents if d.path.startswith(source_path)]
        doc_count = len(docs_from_source)

        # Count chunks for these documents
        chunk_count = 0
        for doc in docs_from_source:
            if doc.id:
                chunks = db.get_chunks_by_document(doc.id)
                chunk_count += len(chunks)

        # Check if source still exists
        path_obj = Path(source_path)
        exists = path_obj.exists()

        result.append({
            "name": source_name,
            "path": source_path,
            "type": "file" if source.get("is_file") else "directory",
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "exists": exists,
            "recursive": source.get("recursive", True),
            "added_at": source.get("added_at"),
        })

    return result


@app.tool
async def get_library_stats(context: Context) -> dict[str, Any]:
    """
    Get overall statistics about the agent's library.

    Shows total documents stored, chunks indexed, and configuration.
    For per-source statistics, use get_library_sources instead.
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


@app.tool
async def get_library_structure(
    context: Context,
    depth: Annotated[int, "How many directory levels to show (1=top level only)"] = 1,
    include_files: Annotated[bool, "Whether to include markdown files in output"] = True,
) -> dict[str, Any]:
    """
    Get the filesystem structure of all library sources.

    Returns the directory tree for each registered source, showing what
    folders and files are available. This helps understand the organization
    of knowledge in the library without reading the actual content.

    Use this to:
    - Explore what topics/categories exist in the library
    - Find the right directory to search or add content to
    - Understand how the library is organized
    """
    sources = _get_sources_config()
    if not sources:
        return {"sources": [], "message": "No sources registered"}

    def get_structure(path: Path, current_depth: int, max_depth: int) -> dict[str, Any]:
        """Recursively build directory structure."""
        if not path.exists():
            return {"error": f"Path does not exist: {path}"}

        if path.is_file():
            return {
                "name": path.name,
                "type": "file",
                "path": str(path),
            }

        structure: dict[str, Any] = {
            "name": path.name,
            "type": "directory",
            "path": str(path),
            "children": [],
        }

        if current_depth >= max_depth:
            # Count children without listing them
            try:
                dirs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
                files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".md"]
                structure["subdirectory_count"] = len(dirs)
                structure["file_count"] = len(files)
            except PermissionError:
                structure["error"] = "Permission denied"
            return structure

        try:
            children = []
            for item in sorted(path.iterdir()):
                # Skip hidden files/directories
                if item.name.startswith("."):
                    continue

                if item.is_dir():
                    child_struct = get_structure(item, current_depth + 1, max_depth)
                    children.append(child_struct)
                elif include_files and item.suffix == ".md":
                    children.append({
                        "name": item.name,
                        "type": "file",
                        "path": str(item),
                    })

            structure["children"] = children
        except PermissionError:
            structure["error"] = "Permission denied"

        return structure

    result_sources = []
    for source in sources:
        source_path = Path(source.get("path", ""))
        source_name = source.get("name", source_path.name)

        if not source_path.exists():
            result_sources.append({
                "name": source_name,
                "path": str(source_path),
                "exists": False,
                "error": "Source path does not exist",
            })
            continue

        structure = get_structure(source_path, 0, depth)
        result_sources.append({
            "name": source_name,
            "path": str(source_path),
            "exists": True,
            "is_file": source.get("is_file", False),
            "structure": structure,
        })

    return {
        "sources": result_sources,
        "total_sources": len(sources),
    }


def _build_directory_doc_index(
    documents: list[Any],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """
    Build indexes for efficient directory-based document lookup.

    Args:
        documents: List of document objects with path and title attributes.

    Returns:
        Tuple of (docs_by_dir, count_by_prefix) for O(1) lookups.
    """
    from collections import defaultdict

    # Index documents by their parent directory
    docs_by_dir: dict[str, list[Any]] = defaultdict(list)
    for doc in documents:
        parent = str(Path(doc.path).parent)
        docs_by_dir[parent].append(doc)

    # Pre-compute counts by path prefix for recursive counts
    count_by_prefix: dict[str, int] = defaultdict(int)
    for doc in documents:
        # Add to count for each parent directory in the path
        path = doc.path
        for prefix_path in [str(Path(path).parents[i]) for i in range(len(Path(path).parents))]:
            count_by_prefix[prefix_path] += 1

    return dict(docs_by_dir), dict(count_by_prefix)


@app.tool
async def get_library_sections(
    context: Context,
    include_file_counts: Annotated[bool, "Include document counts per section"] = True,
) -> dict[str, Any]:
    """
    Get a simplified view of library sections for adding new content.

    THIS IS THE PRIMARY TOOL TO USE BEFORE ADDING CONTENT.

    Returns all available locations where content can be stored, organized
    by source. Each section includes:
    - The full path to use with add_to_library()
    - A description based on existing content
    - Document counts to understand section size

    Use this to:
    1. Discover where different types of content belong
    2. Get the exact path needed for add_to_library(directory=...)
    3. Understand the library's organizational structure

    Example workflow:
    1. Call get_library_sections() to see available locations
    2. Choose the appropriate section based on content type
    3. Call add_to_library(content=..., title=..., directory=<path from step 1>)
    """
    sources = _get_sources_config()
    if not sources:
        return {
            "sections": [],
            "message": "No sources registered. Use 'libr add <path>' to add a source.",
            "default_path": DOCUMENTS_PATH,
        }

    db = get_database()
    all_documents = db.list_documents()

    # Build efficient lookup indexes once
    docs_by_dir, count_by_prefix = _build_directory_doc_index(all_documents)

    sections = []

    for source in sources:
        source_path = Path(source.get("path", ""))
        source_name = source.get("name", source_path.name)
        source_path_str = str(source_path)

        if not source_path.exists():
            sections.append({
                "source": source_name,
                "path": source_path_str,
                "available": False,
                "error": "Source path does not exist",
            })
            continue

        # For file sources, just list the file
        if source.get("is_file"):
            sections.append({
                "source": source_name,
                "path": str(source_path.parent),
                "available": True,
                "type": "file",
                "description": f"Single file: {source_path.name}",
            })
            continue

        # Get top-level directories and their doc counts
        source_sections = []

        # Count docs directly in source root (O(1) lookup)
        root_direct_docs = docs_by_dir.get(source_path_str, [])
        # Count all docs under source (O(1) lookup)
        total_source_docs = count_by_prefix.get(source_path_str, 0)

        source_sections.append({
            "name": source_name + " (root)",
            "path": source_path_str,
            "document_count": len(root_direct_docs) if include_file_counts else None,
            "description": "Root level of this source",
        })

        # Get subdirectories
        try:
            for item in sorted(source_path.iterdir()):
                if item.name.startswith("."):
                    continue
                if not item.is_dir():
                    continue

                item_str = str(item)

                # O(1) count lookup instead of O(n) list comprehension
                dir_doc_count = count_by_prefix.get(item_str, 0)

                # Get sample titles from direct children only
                direct_docs = docs_by_dir.get(item_str, [])
                sample_titles = [d.title for d in direct_docs[:3] if d.title]

                description = f"Contains {dir_doc_count} documents"
                if sample_titles:
                    description += f" (e.g., {', '.join(sample_titles[:2])})"

                # Check for nested subdirectories
                try:
                    subdirs = [
                        p.name for p in item.iterdir() if p.is_dir() and not p.name.startswith(".")
                    ]
                except PermissionError:
                    subdirs = []

                section_info: dict[str, Any] = {
                    "name": item.name,
                    "path": item_str,
                    "description": description,
                }

                if include_file_counts:
                    section_info["document_count"] = dir_doc_count

                if subdirs:
                    section_info["has_subdirectories"] = True
                    section_info["subdirectories"] = subdirs[:10]

                source_sections.append(section_info)

        except PermissionError:
            pass

        sections.append({
            "source": source_name,
            "source_path": source_path_str,
            "available": True,
            "type": "directory",
            "sections": source_sections,
            "total_documents": total_source_docs if include_file_counts else None,
        })

    return {
        "sections": sections,
        "total_sources": len(sources),
        "usage_hint": "Use the 'path' value from any section as the 'directory' parameter in add_to_library()",
    }


@app.tool
async def suggest_library_location(
    context: Context,
    title: Annotated[str, "The title of the content you want to add"],
    content_summary: Annotated[str, "A brief description of what the content is about"] = "",
) -> dict[str, Any]:
    """
    Get suggestions for where to store new content in the library.

    Analyzes the title and optional summary to suggest the best location(s)
    based on existing library organization and similar documents.

    Returns ranked suggestions with paths ready to use with add_to_library().
    """
    sources = _get_sources_config()
    if not sources:
        return {
            "suggestions": [],
            "message": "No sources configured",
            "default_path": DOCUMENTS_PATH,
        }

    # Build search query from title and summary
    search_query = title
    if content_summary:
        search_query = f"{title} {content_summary}"

    # Search for similar content to find where related docs are stored
    # Try hybrid search first, fall back to keyword-only if embeddings fail
    searcher = _get_searcher()
    search_method = "hybrid"
    try:
        similar_results = searcher.search(search_query, limit=10, use_mmr=True)
    except Exception as e:
        # Embedding service unavailable - fall back to keyword search
        logger.warning("Hybrid search failed, falling back to keyword search: %s", e)
        try:
            similar_results = searcher.keyword_search(search_query, limit=10)
            search_method = "keyword"
        except Exception:
            # Even keyword search failed - return source suggestions only
            similar_results = []
            search_method = "none"

    # Analyze where similar documents are located
    location_scores: dict[str, dict[str, Any]] = {}

    for result in similar_results:
        doc_path = Path(result.document_path)
        parent_dir = str(doc_path.parent)

        if parent_dir not in location_scores:
            # Find which source this belongs to
            source = _find_source_for_path(doc_path, sources)
            source_name = source.get("name") if source else "Unknown"

            location_scores[parent_dir] = {
                "path": parent_dir,
                "source": source_name,
                "score": 0,
                "similar_docs": [],
                "breadcrumb": _build_breadcrumb(doc_path.parent, source),
            }

        location_scores[parent_dir]["score"] += result.score
        if len(location_scores[parent_dir]["similar_docs"]) < 3:
            location_scores[parent_dir]["similar_docs"].append({
                "title": Path(result.document_path).stem,
                "relevance": round(result.score, 3),
            })

    # Sort by score and build suggestions
    sorted_locations = sorted(location_scores.values(), key=lambda x: x["score"], reverse=True)

    # Normalize confidence relative to the best score
    max_score = sorted_locations[0]["score"] if sorted_locations else 1.0

    suggestions = []
    for loc in sorted_locations[:5]:
        # Confidence is relative to best match (0.0 to 1.0)
        confidence = loc["score"] / max_score if max_score > 0 else 0.0
        suggestions.append({
            "path": loc["path"],
            "source": loc["source"],
            "confidence": round(confidence, 2),
            "breadcrumb_display": " → ".join(loc["breadcrumb"]),
            "reason": f"Similar to: {', '.join(d['title'] for d in loc['similar_docs'])}",
            "similar_documents": loc["similar_docs"],
        })

    # If no suggestions from search, list available sources
    if not suggestions:
        for source in sources:
            if source.get("is_file"):
                continue
            source_path = source.get("path", "")
            if Path(source_path).exists():
                suggestions.append({
                    "path": source_path,
                    "source": source.get("name", source_path),
                    "confidence": 0.5,
                    "breadcrumb_display": source.get("name", source_path),
                    "reason": "No similar content found - suggesting source roots",
                })

    return {
        "title": title,
        "suggestions": suggestions,
        "search_method": search_method,
        "usage_hint": "Use the suggested 'path' as the 'directory' parameter in add_to_library()",
    }


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    transport_arg = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport_arg not in ("http", "stdio"):
        transport_arg = "stdio"
    app.run(transport=transport_arg, host="127.0.0.1", port=8000)  # type: ignore[arg-type]

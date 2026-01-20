#!/usr/bin/env python3
"""
Librarian CLI - Markdown Document Management System.

A command-line interface for managing, indexing, and searching
markdown documents with vector and full-text search capabilities.

Usage:
    librarian --help
    librarian sources list
    librarian docs list
    librarian docs add <path>
    librarian search "query"
    librarian config show
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

# Suppress verbose logging BEFORE any librarian imports
os.environ.setdefault("LOGURU_LEVEL", "DEBUG")

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Initialize Typer app
app = typer.Typer(
    name="librarian",
    help="Librarian - Markdown Document Management System",
    add_completion=True,
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=True,
    pretty_exceptions_short=True
)

# Sub-commands
sources_app = typer.Typer(help="Manage document sources (directories)", no_args_is_help=True)
docs_app = typer.Typer(help="Manage documents", no_args_is_help=True)
config_app = typer.Typer(help="Configure librarian settings", no_args_is_help=True)

app.add_typer(sources_app, name="sources")
app.add_typer(docs_app, name="docs")
app.add_typer(config_app, name="config")


@app.command("help", hidden=True)
def help_cmd(ctx: typer.Context) -> None:
    """Show this help message."""
    if ctx.parent is not None:
        console.print(ctx.parent.get_help())


console = Console()

# Config file path
CONFIG_DIR = Path.home() / ".librarian"
SOURCES_FILE = CONFIG_DIR / "sources.json"


def _load_sources() -> list[dict[str, Any]]:
    """Load registered sources from config."""
    if not SOURCES_FILE.exists():
        return []
    with open(SOURCES_FILE) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def _save_sources(sources: list[dict[str, Any]]) -> None:
    """Save sources to config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_FILE, "w") as f:
        json.dump(sources, f, indent=2)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _get_config() -> dict[str, Any]:
    """Lazy import of config to avoid early logging."""
    from librarian.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATABASE_PATH,
        DOCUMENTS_PATH,
        EMBEDDING_MODEL,
        HYBRID_ALPHA,
        MMR_LAMBDA,
        SEARCH_LIMIT,
        ensure_directories,
    )

    return {
        "DOCUMENTS_PATH": DOCUMENTS_PATH,
        "DATABASE_PATH": DATABASE_PATH,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SEARCH_LIMIT": SEARCH_LIMIT,
        "MMR_LAMBDA": MMR_LAMBDA,
        "HYBRID_ALPHA": HYBRID_ALPHA,
        "ensure_directories": ensure_directories,
    }


# =============================================================================
# Sources Commands
# =============================================================================


@sources_app.command("list")
def sources_list() -> None:
    """List all registered document sources."""
    sources = _load_sources()

    if not sources:
        rprint(
            Panel(
                "[yellow]No sources registered yet.[/yellow]\n\n"
                "Add a source with: [cyan]librarian sources add <path>[/cyan]",
                title="📂 Document Sources",
            )
        )
        return

    table = Table(title="📂 Document Sources", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Path", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Files", justify="right")

    for source in sources:
        path = Path(source["path"])
        status = "✓ Active" if path.exists() else "✗ Missing"
        file_count = len(list(path.rglob("*.md"))) if path.exists() else 0
        table.add_row(
            source.get("name", path.name),
            str(path),
            source.get("type", "local"),
            status,
            str(file_count),
        )

    console.print(table)


@sources_app.command("add")
def sources_add(
    path: Annotated[str, typer.Argument(help="Path to the directory to add as a source")],
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Custom name for the source")
    ] = None,
    recursive: Annotated[
        bool, typer.Option("--recursive/--no-recursive", "-r", help="Index subdirectories")
    ] = True,
) -> None:
    """Add a new document source directory."""
    source_path = Path(path).resolve()

    if not source_path.exists():
        rprint(f"[red]Error:[/red] Path does not exist: {source_path}")
        raise typer.Exit(1)

    if not source_path.is_dir():
        rprint(f"[red]Error:[/red] Path is not a directory: {source_path}")
        raise typer.Exit(1)

    sources = _load_sources()

    # Check if already exists
    for existing in sources:
        if Path(existing["path"]).resolve() == source_path:
            rprint(f"[yellow]Source already registered:[/yellow] {source_path}")
            return

    source = {
        "name": name or source_path.name,
        "path": str(source_path),
        "type": "local",
        "recursive": recursive,
        "added_at": datetime.now().isoformat(),
    }

    sources.append(source)
    _save_sources(sources)

    # Count markdown files
    md_files = list(source_path.rglob("*.md") if recursive else source_path.glob("*.md"))

    rprint(
        Panel(
            f"[green]✓ Source added successfully![/green]\n\n"
            f"Name: [cyan]{source['name']}[/cyan]\n"
            f"Path: [blue]{source_path}[/blue]\n"
            f"Markdown files found: [yellow]{len(md_files)}[/yellow]\n\n"
            f"Index with: [cyan]librarian docs index[/cyan]",
            title="📂 Source Added",
        )
    )


@sources_app.command("remove")
def sources_remove(
    name_or_path: Annotated[str, typer.Argument(help="Name or path of the source to remove")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Remove a document source."""
    sources = _load_sources()
    to_remove = None

    for source in sources:
        if source.get("name") == name_or_path or source["path"] == name_or_path:
            to_remove = source
            break

    if not to_remove:
        rprint(f"[red]Error:[/red] Source not found: {name_or_path}")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Remove source '{to_remove['name']}'?")
        if not confirm:
            rprint("[yellow]Cancelled.[/yellow]")
            return

    sources.remove(to_remove)
    _save_sources(sources)
    rprint(f"[green]✓ Source removed:[/green] {to_remove['name']}")


def _find_source(name_or_path: str) -> dict | None:
    """Find a source by name or path."""
    sources = _load_sources()
    for s in sources:
        if s.get("name") == name_or_path or s["path"] == name_or_path:
            return s
    return None


def _build_file_tree(source: dict) -> Tree:
    """Build a file tree for a source."""
    source_path = Path(source["path"])
    tree = Tree(f"📂 [bold cyan]{source['name']}[/bold cyan]")

    if not source_path.exists():
        return tree

    recursive = source.get("recursive", True)
    md_files = list(source_path.rglob("*.md") if recursive else source_path.glob("*.md"))

    # Group by directory
    dirs: dict[Path, list[Path]] = {}
    for f in md_files:
        dirs.setdefault(f.parent, []).append(f)

    for dir_path, files in sorted(dirs.items()):
        if dir_path == source_path:
            for f in sorted(files):
                tree.add(f"{f.name}")
        else:
            branch = tree.add(f"{dir_path.relative_to(source_path)}/")
            for f in sorted(files):
                branch.add(f"{f.name}")

    return tree


@sources_app.command("show")
def sources_show(
    name_or_path: Annotated[str, typer.Argument(help="Name or path of the source")],
) -> None:
    """Show details about a specific source."""
    source = _find_source(name_or_path)
    if not source:
        rprint(f"[red]Error:[/red] Source not found: {name_or_path}")
        raise typer.Exit(1)

    tree = _build_file_tree(source)
    console.print(Panel(tree, title=f"Source: {source['name']}"))

    # Show metadata
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Path", source["path"])
    table.add_row("Type", source.get("type", "local"))
    table.add_row("Recursive", str(source.get("recursive", True)))
    table.add_row("Added", source.get("added_at", "Unknown"))

    console.print(table)


# =============================================================================
# Docs Commands
# =============================================================================


@docs_app.command("list")
def docs_list(
    source: Annotated[
        Optional[str], typer.Option("--source", "-s", help="Filter by source name")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum documents to show")] = 50,
) -> None:
    """List all indexed documents."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    from librarian.storage.database import get_database

    db = get_database()
    documents = db.list_documents()

    if source:
        sources = _load_sources()
        source_path = None
        for s in sources:
            if s["name"] == source:
                source_path = s["path"]
                break
        if source_path:
            documents = [d for d in documents if d.path.startswith(source_path)]

    if not documents:
        rprint(
            Panel(
                "[yellow]No documents indexed yet.[/yellow]\n\n"
                "Index documents with: [cyan]librarian docs index[/cyan]",
                title="Documents",
            )
        )
        return

    table = Table(
        title=f"Indexed Documents ({len(documents)} total)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Title", style="green", max_width=40)
    table.add_column("Path", style="blue", max_width=50)
    table.add_column("Updated", style="yellow", width=12)

    for doc in documents[:limit]:
        # Handle updated_at as either datetime or string
        if doc.updated_at:
            if isinstance(doc.updated_at, str):
                updated = doc.updated_at[:10]  # Take YYYY-MM-DD part
            else:
                updated = doc.updated_at.strftime("%Y-%m-%d")
        else:
            updated = "N/A"
        table.add_row(str(doc.id), doc.title or "Untitled", doc.path, updated)

    console.print(table)

    if len(documents) > limit:
        rprint(
            f"[dim]Showing {limit} of {len(documents)} documents. Use --limit to show more.[/dim]"
        )


# Top-level shortcut for listing documents
@app.command("list")
def list_cmd(
    source: Annotated[
        Optional[str], typer.Option("--source", "-s", help="Filter by source name")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum documents to show")] = 50,
) -> None:
    """List all indexed documents (shortcut for 'docs list')."""
    docs_list(source=source, limit=limit)


@docs_app.command("index")
def docs_index(
    source: Annotated[
        Optional[str], typer.Option("--source", "-s", help="Index only a specific source")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force re-index existing documents")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show file paths being indexed")
    ] = False,
) -> None:
    """Index documents from all sources using the server's ingest function."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    sources = _load_sources()
    if not sources:
        rprint("[yellow]No sources registered. Add a source first:[/yellow]")
        rprint("  [cyan]librarian sources add <path>[/cyan]")
        raise typer.Exit(1)

    if source:
        sources = [s for s in sources if s["name"] == source]
        if not sources:
            rprint(f"[red]Source not found:[/red] {source}")
            raise typer.Exit(1)

    # Import and use the server's ingest function (lazy import)
    from librarian.server import ingest_directory as server_ingest

    total_indexed = 0
    total_skipped = 0
    total_errors = 0

    for src in sources:
        src_path = Path(src["path"])
        if not src_path.exists():
            rprint(f"[yellow]Skipping missing source:[/yellow] {src['name']}")
            continue

        rprint(f"[cyan]Indexing {src['name']}...[/cyan]")

        # Use the server's async ingest function (decorated tool function)
        result = _run_async(
            server_ingest(
                context=None,  # type: ignore[arg-type]
                directory=str(src_path),
                recursive=src.get("recursive", True),
                force_reindex=force,
            )
        )

        total_indexed += result.get("indexed", 0)
        total_skipped += result.get("skipped", 0)
        total_errors += len(result.get("errors", []))

        # Print file paths in verbose mode
        if verbose:
            for file_info in result.get("files", []):
                path = file_info.get("path", "")
                status = file_info.get("status", "")
                if status == "created":
                    rprint(f"  [green]+[/green] {path}")
                elif status == "updated":
                    rprint(f"  [yellow]~[/yellow] {path}")
                elif status == "skipped":
                    rprint(f"  [dim]-[/dim] {path}")

        # Show any errors with details
        for err in result.get("errors", []):
            err_path = err.get("path", "unknown")
            err_msg = err.get("error", "Unknown error")
            if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                rprint(f"[yellow]Timeout:[/yellow] {err_path} (file may be on cloud/network)")
            else:
                rprint(f"[red]Error:[/red] {err_path}: {err_msg}")

    status = "complete" if total_errors == 0 else "complete with errors"
    rprint(
        Panel(
            f"[green]Indexing {status}[/green]\n\n"
            f"Indexed: [cyan]{total_indexed}[/cyan]\n"
            f"Skipped: [yellow]{total_skipped}[/yellow]\n"
            f"Errors: [red]{total_errors}[/red]",
            title="Index Results",
        )
    )


@docs_app.command("add")
def docs_add(
    path: Annotated[str, typer.Argument(help="Path to the markdown file to add")],
) -> None:
    """Add and index a single markdown file."""
    cfg = _get_config()
    cfg["ensure_directories"]()
    file_path = Path(path).resolve()

    if not file_path.exists():
        rprint(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    if file_path.suffix != ".md":
        rprint(f"[red]Error:[/red] Not a markdown file: {file_path}")
        raise typer.Exit(1)

    # Use the server's helper function (lazy import)
    from librarian.server import _process_and_index_file

    with console.status(f"Indexing {file_path.name}..."):
        try:
            result = _process_and_index_file(file_path)
            rprint(
                Panel(
                    f"[green]✓ Document indexed![/green]\n\n"
                    f"Title: [cyan]{result['title']}[/cyan]\n"
                    f"Path: [blue]{result['path']}[/blue]\n"
                    f"Chunks: [yellow]{result['chunks']}[/yellow]\n"
                    f"Status: {result['status']}",
                    title="Document Added",
                )
            )
        except Exception as e:
            rprint(f"[red]Error indexing document:[/red] {e}")
            raise typer.Exit(1) from None


@docs_app.command("remove")
def docs_remove(
    path: Annotated[str, typer.Argument(help="Path to the document to remove")],
    delete_file: Annotated[
        bool, typer.Option("--delete-file", "-d", help="Also delete the file")
    ] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Remove a document from the index."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    # Lazy import
    from librarian.server import delete_document as server_delete

    if not force:
        confirm = typer.confirm(f"Remove document at '{path}' from index?")
        if not confirm:
            rprint("[yellow]Cancelled.[/yellow]")
            return

    result = _run_async(server_delete(context=None, path=path, delete_file=delete_file))  # type: ignore[call-arg, arg-type]

    if "error" in result:
        rprint(f"[red]Error:[/red] {result['error']}")
        raise typer.Exit(1)

    rprint(f"[green]✓ Removed from index:[/green] {path}")
    if delete_file and result.get("file_deleted"):
        rprint("[green]✓ File deleted[/green]")


@docs_app.command("show")
def docs_show(
    path: Annotated[str, typer.Argument(help="Path to the document")],
) -> None:
    """Show details about a specific document."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    # Lazy import
    from librarian.server import read_document as server_read

    result = _run_async(server_read(context=None, path=path))  # type: ignore[call-arg, arg-type]

    if "error" in result:
        rprint(f"[red]Error:[/red] {result['error']}")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]{result.get('title', 'Untitled')}[/bold]", title="Document"))

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Path", result.get("path", "N/A"))
    table.add_row("Created", result.get("created_at", "N/A"))
    table.add_row("Updated", result.get("updated_at", "N/A"))
    if result.get("metadata"):
        table.add_row("Metadata", json.dumps(result["metadata"], indent=2))

    console.print(table)

    # Show content preview
    content = result.get("content", "")
    content_preview = content[:500] + "..." if len(content) > 500 else content
    console.print(Panel(content_preview, title="Content Preview"))


# =============================================================================
# Search Command
# =============================================================================


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum results")] = 10,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Search mode: hybrid, vector, keyword")
    ] = "hybrid",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show content")] = False,
) -> None:
    """Search for documents using semantic and keyword search."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    # Lazy import
    from librarian.server import keyword_search, search, vector_search

    with console.status(f"Searching for '{query}'..."):
        if mode == "vector":
            results = _run_async(vector_search(context=None, query=query, limit=limit))  # type: ignore[call-arg, arg-type]
        elif mode == "keyword":
            results = _run_async(keyword_search(context=None, query=query, limit=limit))  # type: ignore[call-arg, arg-type]
        else:
            results = _run_async(search(context=None, query=query, limit=limit, use_mmr=True))  # type: ignore[call-arg, arg-type]

    if not results:
        rprint(Panel("[yellow]No results found.[/yellow]", title="Search Results"))
        return

    rprint(f"\n[bold]Search Results[/bold] - {len(results)} matches for '[green]{query}[/green]'\n")

    home = str(Path.home())

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Path", style="blue")

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        doc_path = result.get("document_path", "Unknown")
        # Shorten path: replace home with ~
        short_path = doc_path.replace(home, "~") if doc_path != "Unknown" else "Unknown"

        table.add_row(
            str(i),
            f"[{score_color}]{score:.3f}[/{score_color}]",
            short_path,
        )

    console.print(table)

    # Show content in verbose mode
    if verbose and results:
        rprint("\n[bold]Content Previews:[/bold]\n")
        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            content = result.get("content", "")[:300]
            content = content.replace("\n", " ").strip()
            if len(result.get("content", "")) > 300:
                content += "..."
            rprint(f"[dim]{i}.[/dim] {content}\n")


# =============================================================================
# Config Commands
# =============================================================================


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = _get_config()

    table = Table(title="Librarian Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value", style="yellow")
    table.add_column("Source", style="dim")

    config_items = [
        ("Documents Path", cfg["DOCUMENTS_PATH"], "DOCUMENTS_PATH"),
        ("Database Path", cfg["DATABASE_PATH"], "DATABASE_PATH"),
        ("Embedding Model", cfg["EMBEDDING_MODEL"], "EMBEDDING_MODEL"),
        ("Chunk Size", str(cfg["CHUNK_SIZE"]), "CHUNK_SIZE"),
        ("Chunk Overlap", str(cfg["CHUNK_OVERLAP"]), "CHUNK_OVERLAP"),
        ("Search Limit", str(cfg["SEARCH_LIMIT"]), "SEARCH_LIMIT"),
        ("MMR Lambda", str(cfg["MMR_LAMBDA"]), "MMR_LAMBDA"),
        ("Hybrid Alpha", str(cfg["HYBRID_ALPHA"]), "HYBRID_ALPHA"),
    ]

    for name, value, env_var in config_items:
        source = "env" if os.environ.get(env_var) else "default"
        table.add_row(name, value, source)

    console.print(table)


@config_app.command("path")
def config_path() -> None:
    """Show configuration file paths."""
    cfg = _get_config()
    rprint(f"Config directory: [cyan]{CONFIG_DIR}[/cyan]")
    rprint(f"Sources file: [cyan]{SOURCES_FILE}[/cyan]")
    rprint(f"Database: [cyan]{cfg['DATABASE_PATH']}[/cyan]")


# =============================================================================
# Stats Command
# =============================================================================


@app.command("stats")
def stats() -> None:
    """Show index statistics."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    # Lazy import
    from librarian.server import get_stats as server_stats

    stats_data = _run_async(server_stats(context=None))  # type: ignore[call-arg, arg-type]

    table = Table(title="Index Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="yellow", justify="right")

    table.add_row("Documents", str(stats_data.get("document_count", 0)))
    table.add_row("Chunks", str(stats_data.get("chunk_count", 0)))

    console.print(table)

    # Show config
    if "config" in stats_data:
        config_table = Table(title="Configuration", show_header=False, box=None)
        config_table.add_column("Key", style="cyan")
        config_table.add_column("Value", style="white")

        for key, value in stats_data["config"].items():
            config_table.add_row(key, str(value))

        console.print(config_table)


# =============================================================================
# Rebuild Command
# =============================================================================


@app.command("rebuild")
def rebuild(
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show file paths being indexed")
    ] = False,
) -> None:
    """
    Rebuild the entire index from scratch.

    Use this when changing embedding models or to fix index corruption.
    Deletes all existing data and re-indexes all sources.
    """
    cfg = _get_config()
    cfg["ensure_directories"]()

    sources = _load_sources()
    if not sources:
        rprint("[yellow]No sources registered. Nothing to rebuild.[/yellow]")
        raise typer.Exit(1)

    # Show warning and get confirmation
    if not confirm:
        rprint(
            Panel(
                "[yellow]Warning: This will delete all indexed data "
                "and rebuild from scratch.[/yellow]\n\n"
                f"Sources to re-index: {len(sources)}\n"
                "This may take a while depending on the number of documents.",
                title="Rebuild Index",
            )
        )
        if not typer.confirm("Continue with rebuild?"):
            rprint("[yellow]Rebuild cancelled.[/yellow]")
            raise typer.Exit(0)

    # Clear the database
    rprint("[cyan]Clearing existing index...[/cyan]")
    from librarian.storage.database import get_database

    db = get_database()
    db.clear_all()
    rprint("[green]Index cleared.[/green]")

    # Re-index all sources with force
    rprint("[cyan]Re-indexing all sources...[/cyan]")
    from librarian.server import ingest_directory as server_ingest

    total_indexed = 0
    total_errors = 0

    for src in sources:
        src_path = Path(src["path"])
        if not src_path.exists():
            rprint(f"[yellow]Skipping missing source:[/yellow] {src['name']}")
            continue

        rprint(f"  Indexing {src['name']}...")

        result = _run_async(
            server_ingest(
                context=None,  # type: ignore[arg-type]
                directory=str(src_path),
                recursive=src.get("recursive", True),
                force_reindex=True,
            )
        )

        total_indexed += result.get("indexed", 0) + result.get("updated", 0)
        total_errors += len(result.get("errors", []))

        # Print file paths in verbose mode
        if verbose:
            for file_info in result.get("files", []):
                path = file_info.get("path", "")
                status = file_info.get("status", "")
                if status == "created":
                    rprint(f"    [green]+[/green] {path}")
                elif status == "updated":
                    rprint(f"    [yellow]~[/yellow] {path}")

        for err in result.get("errors", []):
            err_path = err.get("path", "unknown")
            err_msg = err.get("error", "Unknown error")
            if "timeout" in err_msg.lower():
                rprint(f"  [yellow]Timeout:[/yellow] {err_path}")
            else:
                rprint(f"  [red]Error:[/red] {err_path}: {err_msg}")

    status = "complete" if total_errors == 0 else "complete with errors"
    rprint(
        Panel(
            f"[green]Rebuild {status}[/green]\n\n"
            f"Documents indexed: [cyan]{total_indexed}[/cyan]\n"
            f"Errors: [red]{total_errors}[/red]",
            title="Rebuild Results",
        )
    )


# =============================================================================
# Server Command
# =============================================================================


@app.command("serve")
def serve(
    transport: Annotated[str, typer.Argument(help="Transport: stdio or http")] = "stdio",
    host: Annotated[str, typer.Option("--host", "-h", help="HTTP host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="HTTP port")] = 8000,
) -> None:
    """Start the MCP server."""
    import sys

    from librarian.server import app as mcp_app

    if transport == "stdio":
        # For stdio, redirect all logging to stderr to keep stdout clean for JSON-RPC
        import logging

        logging.basicConfig(
            level=logging.WARNING,
            format="%(name)s: %(message)s",
            stream=sys.stderr,
        )
        # Silence noisy loggers
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
    else:
        rprint(f"[green]Starting HTTP server on {host}:{port}...[/green]")

    mcp_app.run(transport=transport, host=host, port=port)  # type: ignore[arg-type]


# =============================================================================
# Version Command
# =============================================================================


@app.command("version", hidden=True)
def version() -> None:
    """Show version information."""
    rprint("Librarian v0.5.0")


# =============================================================================
# Init Command
# =============================================================================


@app.command("init")
def init(
    path: Annotated[Optional[str], typer.Argument(help="Directory to initialize")] = None,
) -> None:
    """Initialize librarian in a directory."""
    target = Path(path).resolve() if path else Path.cwd()

    # Create config directory
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Add current directory as source if not already
    sources = _load_sources()
    for s in sources:
        if Path(s["path"]).resolve() == target:
            rprint(f"[yellow]Already initialized:[/yellow] {target}")
            return

    source = {
        "name": target.name,
        "path": str(target),
        "type": "local",
        "recursive": True,
        "added_at": datetime.now().isoformat(),
    }
    sources.append(source)
    _save_sources(sources)

    rprint(
        Panel(
            f"[green]✓ Initialized librarian![/green]\n\n"
            f"Directory: [cyan]{target}[/cyan]\n\n"
            f"Next steps:\n"
            f"  1. Index documents: [cyan]librarian docs index[/cyan]\n"
            f"  2. Search: [cyan]librarian search 'your query'[/cyan]",
            title="Librarian Initialized",
        )
    )


if __name__ == "__main__":
    app()

# Librarian

A personal knowledge library for AI agents, built on [Arcade](https://arcade.dev) for the Model Context Protocol (MCP).

## Overview

Librarian provides AI agents with persistent storage for text, documents, and knowledge. Agents can store information and retrieve it later through semantic and keyword search, maintaining context across conversations.

```mermaid
graph LR
    A[Agent Stores Info] --> B[Parser]
    B --> C[Chunker]
    C --> D[Embedder]
    D --> E[(SQLite + vec)]
    F[Agent Queries] --> G[Hybrid Search]
    E --> G
    G --> H[Relevant Context]
```

## Features

- Persistent knowledge storage for AI agents
- SQLite storage with `sqlite-vec` for vector search
- Full-text search using FTS5 with BM25 ranking
- Hybrid search combining semantic and keyword matching
- Max Marginal Relevance (MMR) for diverse results
- Configurable embedding models (local or OpenAI-compatible API)
- Header-aware text chunking with overlap
- Time-bounded search filters
- CLI and MCP server interfaces

## Multi-Modal Support

Librarian supports indexing and searching across multiple file types:

| Asset Type | File Extensions | Features |
|------------|----------------|----------|
| **Text** | `.md`, `.txt` | Frontmatter extraction, header-aware chunking |
| **Code** | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.cpp`, and more | Symbol extraction (classes, functions, methods) |
| **PDF** | `.pdf` | Page-based text extraction |
| **Image** | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Metadata and EXIF extraction, optional OCR |

## Installation

```bash
git clone https://github.com/ArcadeAI/librarian.git
cd librarian
./setup.sh
```

Or install manually:

```bash
uv pip install -e ".[dev]"
```

Optional multi-modal dependencies:

```bash
uv pip install -e ".[pdf]"      # PDF support (pypdf)
uv pip install -e ".[vision]"   # Image support (Pillow)
uv pip install -e ".[all]"      # All optional features
```

## CLI Usage

```bash
# Add files to the library
libr add ~/notes

# Search the library
libr search "machine learning concepts"

# List sources
libr list

# View library statistics
libr index

# Rebuild the index
libr index build
```

## MCP Server

Start the server for AI assistant integration:

```bash
# stdio transport (Claude Desktop, CLI)
libr serve stdio

# HTTP transport (Cursor, VS Code)
libr serve http --port 8000
```

See the [Arcade MCP documentation](https://docs.arcade.dev) for integration details.

### Available Tools

**Core Tools** (always enabled):

| Tool | Description |
|------|-------------|
| `Librarian_SearchLibrary` | Unified search with mode selection (hybrid/semantic/keyword), asset type filtering, and timeframe support |
| `Librarian_AddToLibrary` | Store new content in the library |
| `Librarian_UpdateLibraryDoc` | Update existing content |
| `Librarian_ReadFromLibrary` | Read full document content |
| `Librarian_RemoveFromLibrary` | Remove content from the library |
| `Librarian_ListLibraryContents` | List all stored content |
| `Librarian_IndexDirectoryToLibrary` | Bulk import files from a directory |

**Optional Tools** (enable with `LIBRARIAN_ENABLE_OPTIONAL_TOOLS=true`):

| Tool | Description |
|------|-------------|
| `Librarian_GetLibrarySources` | List sources with document/chunk counts |
| `Librarian_GetLibraryStats` | Overall library statistics |
| `Librarian_GetLibraryStructure` | Filesystem structure of library sources |
| `Librarian_GetLibrarySections` | Simplified view of available storage locations |
| `Librarian_SuggestLibraryLocation` | AI-powered suggestions for where to store content |

## Configuration

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENTS_PATH` | `./documents` | Root directory for files |
| `DATABASE_PATH` | `~/.librarian/index.db` | SQLite database location |
| `EMBEDDING_PROVIDER` | `openai` | `local` or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local model name |
| `OPENAI_API_BASE` | `http://localhost:7171/v1` | OpenAI-compatible API URL |
| `OPENAI_EMBEDDING_MODEL` | `qwen3-embedding-06b` | API model name |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `SEARCH_LIMIT` | `10` | Default results limit |
| `MMR_LAMBDA` | `0.5` | MMR diversity (0=diverse, 1=relevant) |
| `HYBRID_ALPHA` | `0.7` | Vector vs keyword weight (1=vector only) |

## Project Structure

```
librarian/
├── cli.py           # Command-line interface
├── server.py        # MCP server and tool definitions
├── config.py        # Configuration management
├── indexing.py      # Document indexing service
├── types.py         # Shared type definitions
├── storage/
│   ├── database.py  # SQLite operations
│   ├── vector_store.py  # sqlite-vec search
│   └── fts_store.py     # FTS5 search
├── processing/
│   ├── embed/       # Embedding providers
│   ├── parsers/     # Document parsers (md, code, pdf, image)
│   └── transform/   # Text chunking
├── retrieval/
│   └── search.py    # Hybrid search + MMR
└── utils/
    └── timeframe.py # Time filter utilities
```

## Development

```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Run linter
make format     # Format code
make typecheck  # Type checking
make check      # All checks
make evals      # Run evaluations
```

## Resources

- [Arcade.dev](https://arcade.dev) - Build AI-native applications
- [Arcade Documentation](https://docs.arcade.dev) - Integration guides and API reference

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Email: <contact@arcade.dev>
- Website: [arcade.dev](https://arcade.dev)

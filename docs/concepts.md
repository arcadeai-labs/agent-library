# Concepts

Optional reading. The library works fine if you skip this page — but understanding the moving parts helps when you want to tune results.

## Asset types

Agent Library tags every file with an **asset type**:

| Asset type | What's in it | Parser |
|---|---|---|
| `text` | Markdown, plain text | built-in |
| `code` | Source code (Python, JS, TS, Go, Rust, …) | regex-based symbol extractor |
| `pdf` | PDF documents | requires `pypdf` (in `[all]` extras) |
| `image` | PNG, JPG, GIF, WEBP | requires `Pillow` (in `[all]` extras) |

When you search, the asset type is preserved on every result. You can filter on it:

```bash
librarian search "encrypt" --type code
```

## Search modes

Three modes, chosen with `--mode`:

- **`keyword`** — pure full-text search, BM25-ranked. Best for exact phrases or unique tokens. Fast.
- **`semantic`** — pure embedding similarity (cosine distance against a sentence-transformer model). Finds meaning matches even when the wording differs. Slower; loads ~100 MB of model on first use.
- **`hybrid`** *(default)* — runs both and merges. Each modality normalizes its scores to [0, 1]; the merger gives a small overlap bonus to chunks that match across modalities. This is what you want most of the time.

When `ENABLE_CROSS_MODAL_SEARCH=true` (the default), `hybrid` also runs separate embedding models for code (CodeBERT) and images (CLIP) when those extras are installed.

## Chunking

Documents are split into **chunks** before indexing. Each chunk is what the search returns — a passage, not a whole file. This keeps results focused and gives you snippet-level scores instead of file-level.

The chunker is asset-type aware:

- Markdown is split by headers (`H1`/`H2`) and paragraphs
- Code is split by symbol (function, class, method)
- PDFs are split by page
- Images become a single chunk with metadata

## Scoring & MMR

Results are scored 0 → 1. The blend is controlled by two knobs:

- **`HYBRID_ALPHA`** (default `0.7`): in non-cross-modal hybrid, the formula is `alpha * vector_score + (1 - alpha) * keyword_score`. Higher = lean on semantic match more.
- **`MMR_LAMBDA`** (default `0.7`): after blending, **Maximal Marginal Relevance** picks the top-K with a diversity bias. The formula is `lambda * relevance - (1 - lambda) * max_similarity_to_already_selected`. Lower = more diverse top-K (might miss the second-best answer if it looks too much like the first); higher = more relevance-focused.

You can tweak both via environment variables or `librarian config`.

## What lives where

| File | What it is |
|---|---|
| `~/.librarian/index.db` | SQLite database with the FTS5 index, vector embeddings, and document metadata. Survives across sessions. |
| `~/.librarian/sources.json` | The list of registered sources (managed by `librarian add` / `rm`). |
| `~/.librarian/documents/` | Default location for content created via `add_to_library` from inside the MCP server (when no `directory` is given). |

Delete `index.db` and re-run `librarian add ...` to rebuild from scratch.

## MCP under the hood

When an AI assistant calls Agent Library, it speaks the **Model Context Protocol** — a JSON-RPC convention defined by Anthropic. `librarian serve stdio` talks MCP over stdin/stdout; `librarian serve http` talks MCP over HTTP streaming.

The server advertises 9 tools:

| Tool | Purpose |
|---|---|
| `Librarian_SearchLibrary` | The main thing — find content |
| `Librarian_ReadFromLibrary` | Read a full document by path |
| `Librarian_AddToLibrary` | Save new content into the library |
| `Librarian_UpdateLibraryDoc` | Replace a document's content |
| `Librarian_RemoveFromLibrary` | Drop a document from the index |
| `Librarian_ListLibraryContents` | List indexed documents |
| `Librarian_IndexDirectoryToLibrary` | Bulk-index a directory |
| `Librarian_GetLibraryOverview` | Inspect the library (sections / stats / tree) |
| `Librarian_SuggestLibraryLocation` | Recommend where new content belongs |

Each takes typed arguments, returns typed JSON. The MCP host (Claude, Cursor) shows them under the server's name in its tool picker.

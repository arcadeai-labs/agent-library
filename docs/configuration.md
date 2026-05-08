# Configuration

Every knob Agent Library exposes, what it does, what its default is, and **how to change it**. If you only want one or two tweaks, jump straight to [How to change a setting](#how-to-change-a-setting); the rest is reference.

---

## How to change a setting

There are four ways to apply a setting. Pick the one that matches your situation.

=== "`librarian config set` (recommended)"

    The simplest path. Settings are persisted to `~/.librarian/settings.json` and survive across sessions:

    ```bash
    librarian config set EMBEDDING_MODEL "BAAI/bge-base-en-v1.5"
    librarian config set EMBEDDING_DIMENSION 768
    librarian config set HYBRID_ALPHA 0.5
    ```

    Inspect the current state:

    ```bash
    librarian config show       # table of every setting + where each came from
    librarian config get HYBRID_ALPHA
    librarian config path       # show the four config-file paths
    librarian config edit       # open settings.json in your editor
    librarian config reset      # back to defaults
    ```

    Restart `librarian serve` (or your AI client) after changing anything.

=== "From the terminal (one-off)"

    Prefix any `librarian` command with an env var. Useful for trying a setting without committing to it:

    ```bash
    HYBRID_ALPHA=0.5 librarian search "deploy notes"
    ```

=== "Inside Claude Desktop / Cursor / Claude Code"

    The MCP server is a subprocess, so settings the AI host should know about live in the `env` block of the MCP config:

    ```json
    {
      "mcpServers": {
        "librarian": {
          "command": "uvx",
          "args": [
            "--from", "agent-library[all]==0.13.0",
            "librarian", "serve", "stdio"
          ],
          "env": {
            "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
            "EMBEDDING_DIMENSION": "768",
            "MMR_LAMBDA": "0.5"
          }
        }
      }
    }
    ```

    Restart Claude / Cursor after editing.

=== "From a `.env` file"

    Drop a `.env` file in the directory you launch `librarian` from:

    ```
    EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
    HYBRID_ALPHA=0.6
    DATABASE_PATH=/Users/me/work/librarian.db
    ```

    Agent Library reads it automatically on startup.

=== "Permanent shell var"

    Less recommended (env is invisible to GUI apps), but works if everything you care about is terminal-only. Add to `~/.zshrc` or `~/.bashrc`:

    ```bash
    export DATABASE_PATH="$HOME/Documents/librarian.db"
    export EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
    ```

!!! info "Precedence (highest wins)"
    1. Process env vars (`HYBRID_ALPHA=0.5 librarian ...`)
    2. `.env` file in CWD
    3. `librarian config set` (in `~/.librarian/settings.json`)
    4. Built-in defaults

!!! warning "All values must be strings inside JSON"
    JSON env blocks expect `"true"` and `"0.7"`, not `true` or `0.7`. Boolean values that count as "true": `true`, `1`, `yes`, `on` (case-insensitive). Anything else is false.

---

## Storage

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_PATH` | `~/.librarian/index.db` | SQLite file with the FTS index, vectors, and document metadata |
| `DOCUMENTS_PATH` | `./documents` | Default directory used when no `path` is given to a tool |
| `SOURCES_CONFIG_PATH` | `~/.librarian/sources.json` | List of registered sources (managed by `librarian add` / `rm`) |

Set these per-project to keep work and personal libraries separate.

---

## Text embeddings

The library uses an **embedding model** to map text into vectors so semantic search can find meaning matches. The default is fast and small; bigger models give better results at the cost of disk space and CPU.

| Variable | Default | What it does |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` | Either `local` (sentence-transformers on your machine) or `openai` (any OpenAI-compatible endpoint) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | The model to load. See the supported list below |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension. Must match the chosen model — see the supported list |
| `EMBEDDING_QUERY_INSTRUCTION` | `"Given a query, return relevant information from documents."` | Used by instruction-tuned models (E5, BGE) to bias the encoding toward retrieval |

### Supported text models

All of these are loaded via `sentence-transformers`. To switch, set `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` to the matching pair from the table.

| Model | Dim | Size | Notes | HF link |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` *(default)* | 384 | 80 MB | Fast, decent quality, ships everywhere | [→](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `all-mpnet-base-v2` | 768 | 420 MB | The classic sentence-transformers default. Higher quality, ~5× slower | [→](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | BGE small — drop-in replacement for MiniLM with stronger retrieval | [→](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| `BAAI/bge-base-en-v1.5` | 768 | 440 MB | BGE base — what most retrieval benchmarks use | [→](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3 GB | BGE large — best quality of this family, slowest | [→](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| `intfloat/e5-small-v2` | 384 | 130 MB | E5 small — strong retrieval baseline | [→](https://huggingface.co/intfloat/e5-small-v2) |
| `intfloat/e5-base-v2` | 768 | 440 MB | E5 base | [→](https://huggingface.co/intfloat/e5-base-v2) |
| `intfloat/e5-large-v2` | 1024 | 1.3 GB | E5 large | [→](https://huggingface.co/intfloat/e5-large-v2) |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 1.3 GB | Newer model with strong English retrieval scores | [→](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) |

!!! danger "Re-index when you change models"
    Vectors from one model can't be searched with another. After switching `EMBEDDING_MODEL`, delete `~/.librarian/index.db` and re-run `librarian add ...` so your existing content is re-embedded.

### Using OpenAI-compatible APIs

If you'd rather offload embedding to a hosted service (OpenAI, vLLM, llama.cpp's server, etc.), switch the provider:

| Variable | Default | What it does |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` | Set to `openai` |
| `OPENAI_API_BASE` | `http://localhost:7171/v1` | Endpoint URL (point at OpenAI, vLLM, llama.cpp, etc.) |
| `OPENAI_API_KEY` | `not-needed` | Your API key (or `not-needed` for local servers that don't auth) |
| `OPENAI_EMBEDDING_MODEL` | `qwen3-embedding-06b` | Model identifier the endpoint serves |
| `OPENAI_EMBEDDING_DIMENSION` | `1024` | Vector dimension for that model |
| `OPENAI_EMBEDDING_BATCH_SIZE` | `64` | How many texts to embed per API call |

---

## Code embeddings

When `ENABLE_CODE_EMBEDDINGS=true` (the default), source code files are embedded with a code-specific model in addition to the regular text embedder. This makes "find the function that handles retries" work even when "retry" isn't in the comments.

| Variable | Default | What it does |
|---|---|---|
| `ENABLE_CODE_EMBEDDINGS` | `true` | Turn the code path on/off |
| `CODE_EMBEDDING_MODEL` | `microsoft/codebert-base` | The code embedding model |
| `CODE_EMBEDDING_DIMENSION` | `768` | Vector dimension |
| `CODE_EMBEDDING_PROVIDER` | `local` | `local` or `openai` |

### Supported code models

The code path activates when the model name contains `codebert` or `codellama`. Any other model falls back to the regular text path.

| Model | Dim | Size | Notes | HF link |
|---|---|---|---|---|
| `microsoft/codebert-base` *(default)* | 768 | 500 MB | Multi-language, balanced speed/quality | [→](https://huggingface.co/microsoft/codebert-base) |
| `microsoft/graphcodebert-base` | 768 | 500 MB | Better at structural code matches (data flow / graph aware) | [→](https://huggingface.co/microsoft/graphcodebert-base) |

!!! tip "Don't have any code in your library?"
    Set `ENABLE_CODE_EMBEDDINGS=false` to skip loading the model entirely. Saves ~500 MB and a couple seconds at startup.

---

## Vision embeddings

Image files (PNG, JPG, GIF, WEBP) get a separate visual embedding so semantic search works across diagrams and screenshots. This uses CLIP — a model that maps images and text into the same vector space, so a query like "auth flow" finds matching diagrams.

| Variable | Default | What it does |
|---|---|---|
| `ENABLE_VISION_EMBEDDINGS` | `true` | Turn the vision path on/off |
| `VISION_EMBEDDING_MODEL` | `clip-ViT-B-32` | The CLIP-family model |
| `VISION_EMBEDDING_DIMENSION` | `512` | Vector dimension |

### Supported vision models

The vision path activates when the model name contains `clip` or `siglip`.

| Model | Dim | Size | Notes | HF link |
|---|---|---|---|---|
| `clip-ViT-B-32` *(default)* | 512 | 600 MB | Original CLIP base, fast | [→](https://huggingface.co/sentence-transformers/clip-ViT-B-32) |
| `clip-ViT-B-16` | 512 | 600 MB | Higher resolution patches than B-32 — slightly better, slightly slower | [→](https://huggingface.co/sentence-transformers/clip-ViT-B-16) |
| `clip-ViT-L-14` | 768 | 1.7 GB | Large CLIP — best image quality, expensive | [→](https://huggingface.co/sentence-transformers/clip-ViT-L-14) |

!!! tip "Indexing screenshots only?"
    `clip-ViT-B-32` is plenty. The L-14 variants only pay off with photographic content where fine detail matters.

---

## OCR (extracting text from images)

Tesseract-based OCR runs over indexed images so the text inside a screenshot is still searchable.

| Variable | Default | What it does |
|---|---|---|
| `ENABLE_OCR` | `true` | Toggle OCR on indexed images |
| `OCR_LANGUAGE` | `eng` | Tesseract language code(s). Multiple langs use `+` (e.g. `eng+spa`) |
| `OCR_CONFIG` | `--psm 3` | Tesseract page-segmentation mode |
| `OCR_MIN_CONFIDENCE` | `0` | Drop OCR'd text below this confidence (0–100, 0 = no filter) |

OCR requires `tesseract` installed on the system (separate from Python deps): `brew install tesseract` on macOS, `apt install tesseract-ocr` on Debian/Ubuntu.

---

## Image captioning (optional)

When enabled, every indexed image also gets a free-text caption generated by an image-to-text model. Off by default since most users don't need it.

| Variable | Default | What it does |
|---|---|---|
| `IMAGE_GENERATE_CAPTIONS` | `false` | Turn captioning on |
| `IMAGE_CAPTION_MODEL` | `blip-base` | The captioning model |

| Model | Notes | HF link |
|---|---|---|
| `Salesforce/blip-image-captioning-base` *(default — `blip-base` is the short alias)* | BLIP base; fast, decent captions | [→](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| `Salesforce/blip-image-captioning-large` | BLIP large; slower, better captions | [→](https://huggingface.co/Salesforce/blip-image-captioning-large) |

---

## Chunking

How documents get split into searchable chunks before indexing.

| Variable | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | `512` | Target chunk length in tokens (≈ words) |
| `CHUNK_OVERLAP` | `50` | Tokens of overlap between adjacent chunks (preserves context across boundaries) |
| `MIN_CHUNK_SIZE` | `50` | Drop chunks shorter than this |
| `CODE_CHUNK_STRATEGY` | `code_blocks` | `code_blocks` (split by function/class) or `fixed` (fixed size) |
| `CODE_INCLUDE_CONTEXT` | `true` | Include surrounding lines as context when chunking code |
| `CODE_CONTEXT_LINES` | `5` | How many context lines on each side |
| `PDF_CHUNK_STRATEGY` | `pages` | `pages` (one chunk per page) or `sections` (split by headings) |

!!! tip "Larger chunks = more context per result, fewer results overall"
    Bumping `CHUNK_SIZE` to 1024 makes each result longer but reduces the total number of chunks. Good for technical docs where context matters; bad for short notes where you want fine-grained matching.

---

## Search behavior

| Variable | Default | What it does |
|---|---|---|
| `SEARCH_LIMIT` | `10` | Default result count |
| `HYBRID_ALPHA` | `0.7` | In hybrid mode, the blend: `alpha · vector_score + (1 - alpha) · keyword_score`. Higher = more semantic |
| `MMR_LAMBDA` | `0.7` | Maximal Marginal Relevance: `lambda · relevance - (1 - lambda) · max_similarity_to_already_picked`. Higher = relevance-focused, lower = diverse |
| `ENABLE_CROSS_MODAL_SEARCH` | `true` | Search text + code + image vectors in parallel and merge fairly |
| `CROSS_MODAL_SIMILARITY_THRESHOLD` | `0.7` | Drop cross-modal matches below this similarity |
| `MODALITY_WEIGHT_TEXT` | `1.0` | Per-modality weight in the cross-modal merge |
| `MODALITY_WEIGHT_CODE` | `1.0` | |
| `MODALITY_WEIGHT_VISION` | `1.0` | |
| `MODALITY_WEIGHT_FTS` | `1.0` | Keyword search modality weight |

!!! tip "Tuning the search"
    - **Results feel scattered?** Lower `MMR_LAMBDA` toward 0.3 to push for more diverse top-K, or raise it toward 0.9 to lock in on the most relevant.
    - **Hybrid is missing exact-keyword matches?** Lower `HYBRID_ALPHA` toward 0.3 to weight keyword score higher.
    - **Code matches dominating text matches?** Lower `MODALITY_WEIGHT_CODE` to 0.5.

---

## Codebase indexing

| Variable | Default | What it does |
|---|---|---|
| `CODEBASE_AUTO_DETECT` | `true` | Detect language from file extensions automatically |
| `CODEBASE_INDEX_TESTS` | `true` | Include `tests/` and `*_test.*` files |
| `CODEBASE_MAX_FILE_SIZE_KB` | `500` | Skip files larger than this |
| `DEFAULT_ASSET_TYPES` | `text,code` | Comma-separated asset types to index by default |

---

## PDF processing

| Variable | Default | What it does |
|---|---|---|
| `ENABLE_PDF_PROCESSING` | `true` | Toggle PDF parsing |
| `PDF_OCR_ENABLED` | `false` | Run OCR on PDF pages (slow; only useful for scanned PDFs without an embedded text layer) |

---

## Server

| Variable | Default | What it does |
|---|---|---|
| `LIBRARIAN_HOST` | `127.0.0.1` | HTTP transport bind address |
| `LIBRARIAN_PORT` | `8000` | HTTP transport port |
| `LIBRARIAN_ENABLE_OPTIONAL_TOOLS` | `true` | Whether `get_library_overview` and `suggest_library_location` are advertised |

---

## Tool behavior (advanced)

| Variable | Default | What it does |
|---|---|---|
| `TOOL_SEARCH_DEFAULT_LIMIT` | `10` | Default `limit` arg when the AI calls `search_library` without specifying |
| `TOOL_MAX_CONTEXT_LINES` | `10` | Max lines of context surfaced in tool output |
| `CODE_MAX_DEPENDENCY_DEPTH` | `3` | Max depth for code dependency walks |
| `CODE_MAX_REFERENCES` | `50` | Max references returned per query |
| `INDEX_POLL_INTERVAL` | `60.0` | Seconds between background re-index polls |
| `INDEX_START_DELAY` | `5.0` | Seconds before the first background poll on startup |

---

## Putting it together — a real example

Say you want a project-specific library that uses a stronger embedder, weights keyword matches higher than the default, and lives next to your code rather than in `~/.librarian/`. Add this to your Cursor `mcp.json` or Claude Desktop config:

```json
{
  "mcpServers": {
    "librarian": {
      "command": "uvx",
      "args": [
        "--from", "agent-library[all]==0.13.0",
        "librarian", "serve", "stdio"
      ],
      "env": {
        "DATABASE_PATH": "${workspaceFolder}/.librarian/index.db",
        "DOCUMENTS_PATH": "${workspaceFolder}",
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        "EMBEDDING_DIMENSION": "768",
        "HYBRID_ALPHA": "0.5",
        "ENABLE_VISION_EMBEDDINGS": "false"
      }
    }
  }
}
```

After saving, reload Cursor (or restart Claude). Index once with `librarian add .` — and your AI now searches that project's content with the stronger model.

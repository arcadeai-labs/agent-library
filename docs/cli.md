# CLI reference

Every command Agent Library exposes, with copy-paste examples. The binary name is `librarian` (with `libr` as a shorter alias). Examples below use `librarian`.

!!! note "Reminder"
    These all assume you've set up the `librarian` alias from the [Quickstart](quickstart.md). If you haven't, prepend each command with:

    ```
    uvx --from "agent-library[all]==0.13.0"
    ```

---

## `librarian add`

Add a file or directory as a source and index its contents.

```bash
librarian add ~/notes
```

**Options:**

| Flag | What it does |
|---|---|
| `-n NAME` / `--name` | Give the source a friendly name (defaults to the directory name) |
| `-d N` / `--depth` | Limit recursion. `0` = the directory itself only. Default: unlimited |
| `-p PATTERN` / `--pattern` | Glob filter (e.g. `'notes/*.md'`) |

```bash
# Index just the top-level files of a folder
librarian add ~/notes --depth 0

# Index only Python files in a project
librarian add ~/code/myproject --pattern '**/*.py'
```

---

## `librarian list`

Show every source you've added, with document counts.

```bash
librarian list
```

**Options:**

| Flag | What it does |
|---|---|
| `-a` / `--all` | Include sources marked hidden/test |
| `--json` | Output as JSON (good for scripting) |

---

## `librarian rm`

Remove a source from the index. Files on disk are **not** deleted — only the database entries.

```bash
librarian rm notes
```

**Options:**

| Flag | What it does |
|---|---|
| `-f` / `--force` | Skip the confirmation prompt |
| `--path PATH` | Disambiguate if two sources share the same name |

---

## `librarian search`

Search across everything you've indexed.

```bash
librarian search "retry policy"
```

**Options:**

| Flag | What it does |
|---|---|
| `-l N` / `--limit` | Max results (default 10) |
| `-m MODE` / `--mode` | `hybrid` (default), `semantic`, `vector` (alias for semantic), or `keyword` |
| `-s NAME` / `--source` | Search within a single source |
| `-t TIMEFRAME` / `--timeframe` | `today`, `yesterday`, `week`, `month`, `year` |
| `-f FORMAT` / `--format` | Output as `table` (default), `json`, or `paths` |
| `-v` / `--verbose` | Include the matched content snippet inline |
| `-o` / `--open` | Open the top result in your editor |
| `-c` / `--copy` | Copy the top result's content to clipboard |
| `--code` | Search code files only |
| `--images` | Search images only (uses CLIP if vision is installed) |
| `--type TYPE` | Filter by asset type: `text`, `code`, `pdf`, `image` |

```bash
# Find conceptual matches, not just keyword
librarian search "graceful degradation" --mode semantic

# Just the file paths, one per line, useful for piping
librarian search "deploy" --format paths | xargs cat | less

# Things from this week, code only
librarian search "TODO" --timeframe week --code
```

---

## `librarian serve`

Start the MCP server. This is what your AI assistant invokes — you usually configure it once in Claude/Cursor and never run it by hand. But it's available for testing.

```bash
librarian serve stdio    # for Claude Desktop, Claude Code, Cursor
librarian serve http --port 7878   # for HTTP-based MCP clients
```

**Options:**

| Flag | What it does |
|---|---|
| `--host HOST` / `-h` | HTTP-only: bind address |
| `--port N` / `-p` | HTTP-only: port number |
| `--log-level LEVEL` | `debug`, `info`, `warning` (default), or `error` |

---

## Subcommand groups

Three command groups bundle less-common operations:

```bash
librarian index --help    # index inspection / rebuild operations
librarian docs --help     # per-document operations (read, update, remove single docs)
librarian config --help   # show or change configuration values
```

These are in the help output but used less often than the top-level commands above.

---

## Environment variables

Override defaults without editing config files:

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_PATH` | `~/.librarian/index.db` | Where the search index is stored |
| `DOCUMENTS_PATH` | `./documents` | Default directory used when no `path` is given |
| `EMBEDDING_PROVIDER` | `local` | Switch to `openai` for hosted embeddings |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `HYBRID_ALPHA` | `0.7` | Vector ↔ keyword blend (0 = keyword only, 1 = vector only) |
| `MMR_LAMBDA` | `0.7` | Relevance ↔ diversity (0 = max diversity, 1 = max relevance) |
| `SEARCH_LIMIT` | `10` | Default result count |

```bash
# One-off override for a single search
DATABASE_PATH=/tmp/test.db librarian search "anything"
```

For permanent overrides, add them to your shell profile or a `.env` file in the directory you launch `librarian` from.

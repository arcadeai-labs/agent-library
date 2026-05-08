# Quickstart

Five minutes from zero to "my AI can search my notes". You'll do all of this from your terminal.

!!! note "Before you start"
    Make sure `uv --version` works in your terminal. If not, do **[Install uv](install.md)** first (one command).

## 1. Run Agent Library for the first time

Paste this into your terminal:

```bash
uvx --from "agent-library[all]==0.13.0" librarian --help
```

The first time you run this it'll spend a couple minutes downloading the Python package and its language models (~2 GB total). You only pay this cost **once** — `uv` caches everything afterward.

When it finishes you'll see a help menu listing commands like `add`, `search`, `serve`. That confirms it's working.

!!! tip "Make this command shorter"
    Typing `uvx --from "agent-library[all]==0.13.0" librarian` every time is annoying. Add an alias to your shell:

    === "macOS / Linux (zsh)"

        ```bash
        echo 'alias librarian="uvx --from \"agent-library[all]==0.13.0\" librarian"' >> ~/.zshrc
        source ~/.zshrc
        ```

    === "macOS / Linux (bash)"

        ```bash
        echo 'alias librarian="uvx --from \"agent-library[all]==0.13.0\" librarian"' >> ~/.bashrc
        source ~/.bashrc
        ```

    From here on, this guide assumes you have the `librarian` alias. If you didn't set one, just substitute the long form.

## 2. Index a folder of notes

Pick any folder that has text-like files — notes, markdown, PDFs, code. Let's say it's `~/notes/`.

```bash
librarian add ~/notes
```

Agent Library walks the folder, parses each supported file (Markdown, code in 18 languages, PDFs, images), splits the contents into searchable chunks, and stores them in `~/.librarian/index.db`. It'll print a progress summary at the end.

!!! note "Supported file types"
    Out of the box: Markdown (`.md`), text (`.txt`), 18 programming languages (`.py`, `.js`, `.ts`, `.go`, `.rs`, etc.), PDFs, and common image formats (PNG, JPG). Other file types are skipped.

## 3. Search it

```bash
librarian search "what was that idea about retry policy?"
```

Three results print, ranked by how well they match. Each has a path, a snippet, and a relevance score from 0 to 1.

By default this is a **hybrid search** — it combines keyword matching with semantic understanding, so "retry policy" finds notes that talk about "exponential backoff" too.

Two other modes are available when you want them:

```bash
librarian search "exact phrase here" --mode keyword   # exact-match only
librarian search "this concept"      --mode semantic  # meaning-based only
```

## 4. List what you've indexed

```bash
librarian list
```

Shows every source you've added (folders or single files) with document counts.

---

## You're set

The library now has your stuff. Next: tell your AI assistant about it.

- [Set up Claude Desktop →](integrations/claude-desktop.md)
- [Set up Claude Code →](integrations/claude-code.md)
- [Set up Cursor →](integrations/cursor.md)

If something didn't work, head to [Troubleshooting](troubleshooting.md).

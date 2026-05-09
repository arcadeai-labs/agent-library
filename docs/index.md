# Agent Library

**A personal knowledge library for your AI assistant.** Drop your notes, code, PDFs, and screenshots into it once. From then on, Claude (or Cursor, or any other MCP-compatible AI) can search and read your stuff to answer your questions.

You don't need to know what an embedding is. You don't need to know what MCP is. You just need to follow three steps.

---

## What you can do with it

- **Ask Claude about anything in your notes.** "What did I write about the API redesign last month?" — Claude searches your library, finds the relevant note, and reads it back.
- **Have your AI work over your codebase.** Index a project folder once. Cursor or Claude Code can then search the code by meaning ("find the function that handles retry logic") instead of just by filename.
- **Search PDFs and screenshots alongside your notes.** Indexes contracts, papers, diagrams. The library treats them like first-class search results.

It runs entirely on your machine. Nothing leaves your laptop.

---

## How to get started

Three pages, in order:

1. **[Install uv](install.md)** — `uv` is the tool that runs Agent Library. One copy-paste command.
2. **[Quickstart](quickstart.md)** — index a folder and search it from your terminal. Verify it works.
3. **Connect it to your AI** — pick the one you use:
    - **[Claude Desktop](integrations/claude-desktop.md)** — the regular Claude app
    - **[Claude Code](integrations/claude-code.md)** — the terminal-based Claude
    - **[Cursor](integrations/cursor.md)** — the IDE

After that:

- **[CLI reference](cli.md)** — every `librarian` command with a copy-paste example
- **[Concepts](concepts.md)** — what's actually happening under the hood (optional reading)
- **[Troubleshooting](troubleshooting.md)** — when something doesn't work

---

## Why "Agent Library"?

Most AI assistants forget your work the moment a conversation ends. Agent Library is the long-lived shelf they can reach for. You write a note once; six conversations from now your assistant still finds it.

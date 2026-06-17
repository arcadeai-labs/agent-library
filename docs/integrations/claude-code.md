# Claude Code (CLI)

Claude Code is the terminal-based version of Claude. It uses the same MCP server interface as Claude Desktop, but you configure it from the shell.

## Prerequisites

- You ran the [Quickstart](../quickstart.md) and have at least one folder indexed.
- Claude Code is installed: <https://docs.anthropic.com/claude-code>.

## Add Agent Library

Claude Code has a one-line command to register an MCP server. The `--` separator tells Claude where the command-and-args to invoke begin:

```bash
claude mcp add librarian -- uvx --from "agent-library[all]==0.13.0" librarian serve stdio
```

That writes the entry into `~/.claude.json` (or your project-local equivalent depending on `--scope`). Verify:

```bash
claude mcp list
```

You should see `librarian` listed.

!!! tip "Editing the JSON directly"
    If you'd rather edit `~/.claude/settings.json` by hand, the `mcpServers` block looks the same as the Claude Desktop one:

    ```json
    {
      "mcpServers": {
        "librarian": {
          "command": "uvx",
          "args": [
            "--from", "agent-library[all]==0.13.0",
            "librarian", "serve", "stdio"
          ]
        }
      }
    }
    ```

??? tip "Already ran `uv tool install`?"
    Claude Code is a terminal app, so it inherits your shell `PATH`. That means once you've done `uv tool install`, the simpler form works too:

    ```json
    "librarian": {
      "command": "librarian",
      "args": ["serve", "stdio"]
    }
    ```

    `uvx` is still our default in the docs because it's hermetic and doesn't depend on the install state.

## Use it

Start (or restart) Claude Code. In your conversation:

> *"Search my library for the retry policy notes."*

The first response triggers the MCP tool. You'll be prompted to allow `Librarian_SearchLibrary` — accept it once and Claude Code remembers.

## Scope: local, project, or user

`claude mcp add` accepts a `--scope` flag (default: `local`):

| Scope | Where it's stored | When you'd use it |
|---|---|---|
| `local` *(default)* | `.claude.json` for the current project — only you see it | Trying things out, personal experiments |
| `project` | `.mcp.json` at the repo root, checked into git | Sharing a server config with the whole team |
| `user` | `~/.claude.json` — visible from every directory | One library you use across all your projects |

For a personal library available everywhere:

```bash
claude mcp add librarian --scope user -- uvx --from "agent-library[all]==0.13.0" librarian serve stdio
```

For a project-specific library with its own database:

```bash
claude mcp add librarian \
  -e "DATABASE_PATH=$(pwd)/.librarian/index.db" \
  -e "DOCUMENTS_PATH=$(pwd)" \
  -- uvx --from "agent-library[all]==0.13.0" librarian serve stdio
```

Pass `-e KEY=VALUE` once per env var (it can be repeated).

## Removing it

```bash
claude mcp remove librarian
```

---

Problems? See [Troubleshooting](../troubleshooting.md).

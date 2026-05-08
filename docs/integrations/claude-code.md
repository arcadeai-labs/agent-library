# Claude Code (CLI)

Claude Code is the terminal-based version of Claude. It uses the same MCP server interface as Claude Desktop, but you configure it from the shell.

## Prerequisites

- You ran the [Quickstart](../quickstart.md) and have at least one folder indexed.
- Claude Code is installed: <https://docs.anthropic.com/claude-code>.

## Add Agent Library

Claude Code has a one-line command to register an MCP server. From any directory:

```bash
claude mcp add librarian \
  --command "uvx" \
  --args "--from" --args "agent-library[all]==0.13.0" \
       --args "librarian" --args "serve" --args "stdio"
```

That writes the entry into `~/.claude/settings.json` (or your project-local equivalent). Verify:

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

## Use it

Start (or restart) Claude Code. In your conversation:

> *"Search my library for the retry policy notes."*

The first response triggers the MCP tool. You'll be prompted to allow `Librarian_SearchLibrary` — accept it once and Claude Code remembers.

## Project-scoped vs. user-scoped

By default the entry above goes into your **user-level** Claude config so it's available from every directory. If you'd rather have a per-project Agent Library (different index for different projects), pass `--scope local`:

```bash
claude mcp add librarian --scope local \
  --command "uvx" \
  --args "--from" --args "agent-library[all]==0.13.0" \
       --args "librarian" --args "serve" --args "stdio"
```

That writes to `.claude/settings.json` next to your project, and you can point at a project-specific database:

```bash
claude mcp add librarian --scope local \
  --command "uvx" \
  --args "--from" --args "agent-library[all]==0.13.0" \
       --args "librarian" --args "serve" --args "stdio" \
  --env "DATABASE_PATH=$(pwd)/.librarian/index.db" \
  --env "DOCUMENTS_PATH=$(pwd)"
```

## Removing it

```bash
claude mcp remove librarian
```

---

Problems? See [Troubleshooting](../troubleshooting.md).

# Cursor

Cursor supports MCP servers natively as of late 2024. Hooking up Agent Library lets Cursor search your notes and external knowledge alongside whatever's open in your editor.

## Prerequisites

- You ran the [Quickstart](../quickstart.md) and have at least one folder indexed.
- Cursor 0.45 or newer (MCP support shipped in that release).

## 1. Open the MCP settings

In Cursor:

1. Open the command palette: <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> (or <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> on Windows/Linux).
2. Type **"MCP"** and pick **"Cursor: Open MCP Settings"** (or **"Cursor: Edit MCP Servers"** in newer builds).

That opens `~/.cursor/mcp.json`. If the file is empty, Cursor will pre-fill the skeleton.

## 2. Add the Agent Library entry

If the file is empty or freshly skeletoned, paste this:

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

If you already have other MCP servers in there, add `"librarian"` as another key under `mcpServers`. Save the file.

??? tip "Already ran `uv tool install`?"
    Cursor inherits more of your environment than Claude Desktop, so the simpler form usually works once you've installed the binary globally:

    ```json
    "librarian": {
      "command": "librarian",
      "args": ["serve", "stdio"]
    }
    ```

    If Cursor reports "command not found", switch back to the `uvx` form above (or use the absolute path from `which librarian`).

## 3. Reload Cursor

Either fully quit and reopen Cursor, or run **"Developer: Reload Window"** from the command palette. Cursor re-reads `mcp.json` on each reload.

## 4. Try it

Open the Cursor chat panel (<kbd>Cmd</kbd>+<kbd>L</kbd>) and ask:

> *"Search my library for retry policy notes."*

Cursor's agent will call into the librarian server. The first time, you'll see a permission prompt — accept it.

You can confirm the connection from the **MCP** tab in Cursor's settings: librarian should show as **Connected** with a green dot.

---

## Project-specific libraries

Cursor also supports a project-local `mcp.json` at the workspace root. Useful when each codebase has its own indexed knowledge:

```bash
mkdir -p .cursor && cat > .cursor/mcp.json <<'EOF'
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
        "DOCUMENTS_PATH": "${workspaceFolder}"
      }
    }
  }
}
EOF
```

The `${workspaceFolder}` placeholder is expanded by Cursor at launch.

---

Problems? See [Troubleshooting](../troubleshooting.md).

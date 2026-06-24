# Claude Desktop

Connect your Agent Library to the Claude Desktop app so Claude can search your notes during chats.

## Prerequisites

- You ran the [Quickstart](../quickstart.md) and have at least one folder indexed.
- Claude Desktop is installed: <https://claude.ai/download>.

## 1. Find the config file

Claude Desktop reads its tool list from a single JSON file. The path depends on your OS:

=== "macOS"

    ```
    ~/Library/Application Support/Claude/claude_desktop_config.json
    ```

    Open it in TextEdit or your editor of choice:

    ```bash
    open -a TextEdit "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    ```

    If TextEdit complains the file doesn't exist, create it first:

    ```bash
    mkdir -p "$HOME/Library/Application Support/Claude"
    touch "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    ```

=== "Windows"

    ```
    %APPDATA%\Claude\claude_desktop_config.json
    ```

    Open it in Notepad:

    ```powershell
    notepad "$env:APPDATA\Claude\claude_desktop_config.json"
    ```

=== "Linux"

    ```
    ~/.config/Claude/claude_desktop_config.json
    ```

## 2. Add the Agent Library entry

If the file is empty, paste this whole block in:

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

If the file already has content, just add the `"librarian": { ... }` block inside the existing `mcpServers` object. The full file should look like a valid JSON document with `mcpServers` at the top level.

Save the file.

??? tip "Already ran `uv tool install`?"
    If you went through the [Quickstart](../quickstart.md) and installed Agent Library globally with `uv tool install`, you can swap the `uvx` form for the installed binary directly. **You'll need to use the absolute path** because Claude Desktop is launched by macOS's `launchd` and doesn't inherit your terminal's `PATH`:

    ```json
    "librarian": {
      "command": "/Users/YOUR-USERNAME/.local/bin/librarian",
      "args": ["serve", "stdio"]
    }
    ```

    Find your absolute path with `which librarian` in the terminal. The `uvx` form doesn't have this caveat, which is why we recommend it as the default.

## 3. Restart Claude Desktop

Quit Claude entirely (right-click the dock icon → Quit, or <kbd>Cmd</kbd>+<kbd>Q</kbd>) and reopen it. Claude only re-reads the config on launch.

## 4. Try it

Start a new chat with Claude and ask something that requires your notes:

> *"Search my library for notes about retry policy."*

Claude will call the `Librarian_SearchLibrary` tool. The first time, you'll see a permission prompt — click **Allow**. The results appear inline.

You can verify the connection by checking the bottom of the Claude window — when an MCP server is connected, a small plug icon shows "1 server connected".

---

## What if something doesn't work?

!!! warning "Claude says it has no tools"
    The most common cause is that Claude was already running when you saved the config. Quit it completely (<kbd>Cmd</kbd>+<kbd>Q</kbd>, not just close window) and reopen.

!!! warning "Claude says the tool errored out"
    Open `~/Library/Logs/Claude/mcp-server-librarian.log` (macOS) or the equivalent on your OS. The MCP server starts without loading the local ML stack, but the first semantic search may still download and load the text embedding model. Run the model warmup once at the terminal first:

    ```bash
    librarian config models
    ```

    Then restart Claude.

!!! tip "Want a custom storage location?"
    Add an `env` block:

    ```json
    "librarian": {
      "command": "uvx",
      "args": ["--from", "agent-library[all]==0.13.0", "librarian", "serve", "stdio"],
      "env": {
        "DATABASE_PATH": "/Users/me/work/librarian.db",
        "DOCUMENTS_PATH": "/Users/me/work/notes"
      }
    }
    ```

More problems? [Troubleshooting →](../troubleshooting.md)

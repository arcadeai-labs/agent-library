# Troubleshooting

The most common things that go wrong, and how to fix them.

## `uv: command not found`

The installer dropped `uv` into `~/.local/bin` (Linux/macOS) or `~/.cargo/bin` (older versions), but your terminal hasn't picked up the change yet.

**Fix:** Close your terminal completely and reopen it. If that doesn't help, manually add the install dir to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add that line to `~/.zshrc` or `~/.bashrc` to make it stick.

## First run is taking forever

The first `uv tool install "agent-library[all]==0.13.0"` (or `uvx ...` invocation) downloads:

- The Agent Library package
- `sentence-transformers` and `torch` (~500 MB)
- Optional vision/code models on first search (~1 GB)

Plan on 2–5 minutes the first time. Subsequent runs use the cached install and start in under a second.

If it seems stuck, run with verbose output:

```bash
uv tool install --verbose "agent-library[all]==0.13.0"
```

## Claude says it doesn't have any tools

Three things to check:

1. **Did you fully quit Claude before reopening?** A cold restart is required after editing `claude_desktop_config.json`. Close menus aren't enough — use <kbd>Cmd</kbd>+<kbd>Q</kbd> on macOS.
2. **Is the JSON valid?** Paste the whole file into <https://jsonlint.com/> to confirm. A common mistake is leaving a trailing comma after the last entry.
3. **Did the MCP subprocess crash?** Check `~/Library/Logs/Claude/mcp-server-librarian.log` (macOS) for stack traces. If you see "command not found: uvx", the issue is your PATH — see the section above.

## "First time" timeouts inside Claude

Claude Desktop has an internal timeout for MCP server startup. If your first run hits the ~2-minute model download, Claude may give up before the server is ready.

**Fix:** Warm the cache once at the terminal first:

```bash
uv tool install "agent-library[all]==0.13.0"
librarian --help
```

Then restart Claude. The next launch will reuse the cached install.

## "ModuleNotFoundError: No module named 'pypdf'" / 'PIL'

You installed without the `[all]` extras. PDFs and images are skipped silently when their parsers aren't available.

**Fix:** Reinstall with all extras:

```bash
uv tool install --reinstall "agent-library[all]==0.13.0"
```

Or, in your MCP config, change `"agent-library==0.13.0"` to `"agent-library[all]==0.13.0"`.

## Search returns nothing

A few likely causes:

- **You haven't indexed anything yet.** Run `librarian list` — if it's empty, run `librarian add <some-folder>` first.
- **Your query is too narrow.** Try `--mode hybrid` (the default — combines semantic and keyword) and a shorter query. Also try `--type text` if you're certain the content is text-based.
- **The index is stale.** Re-run `librarian add <path>` to refresh. New files are picked up automatically; modified ones are re-indexed.

## Search results look weird / irrelevant

Try a different `--mode`:

- `keyword` is best for exact-phrase searches.
- `semantic` is best when you want meaning matches but no token overlap.
- `hybrid` (default) blends the two.

If hybrid results feel diluted, you can also tune `MMR_LAMBDA`:

```bash
MMR_LAMBDA=0.9 librarian search "your query"   # heavily favor relevance over diversity
MMR_LAMBDA=0.3 librarian search "your query"   # heavily favor diversity (different docs)
```

## I want to wipe everything and start over

```bash
rm -rf ~/.librarian
uv tool uninstall agent-library
```

Then reinstall via Quickstart.

## I'm hitting permission errors writing to `~/.librarian`

Override the location with an env var:

```bash
DATABASE_PATH="$HOME/Documents/librarian.db" librarian add ~/notes
```

Set it permanently in your shell profile:

```bash
echo 'export DATABASE_PATH="$HOME/Documents/librarian.db"' >> ~/.zshrc
```

---

Still stuck? Open an issue at <https://github.com/ArcadeAI/agent-library/issues> with:

1. What you ran
2. What you expected
3. What happened (full error output)
4. `librarian --version` output (run after `uv tool install agent-library`)

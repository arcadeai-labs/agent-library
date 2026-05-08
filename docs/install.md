# Installing uv

`uv` is the tool that runs Agent Library. It's a fast Python package manager — think of it as the "app store" we'll fetch Agent Library from.

You only have to do this once.

## macOS

Open the **Terminal** app (Spotlight → "Terminal") and paste this:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Press <kbd>Enter</kbd>. After it finishes (a few seconds), close and reopen Terminal so the new command is on your path.

Verify it worked:

```bash
uv --version
```

You should see something like `uv 0.5.x`. If you do, you're done — go to **[Quickstart](quickstart.md)**.

??? tip "Prefer Homebrew?"
    If you already use Homebrew:

    ```bash
    brew install uv
    ```

## Windows

Open **PowerShell** (Start menu → "PowerShell") and paste this:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

## Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

---

## Troubleshooting

!!! warning "`uv: command not found`"
    The installer sets up your `PATH` in a shell profile that only reloads on a new shell. **Close your terminal entirely and reopen it**, then try `uv --version` again. If it still fails, run `source ~/.zshrc` (macOS default) or `source ~/.bashrc` (Linux) and try once more.

!!! warning "`Could not connect to astral.sh`"
    Corporate networks sometimes block the install script. If you have Python installed already, you can fall back to:

    ```bash
    pip install --user uv
    ```

When `uv --version` prints a version number, head to **[Quickstart](quickstart.md)**.

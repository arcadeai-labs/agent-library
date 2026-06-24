"""Import boundary tests for lightweight MCP startup."""

import json
import subprocess
import sys


def test_librarian_server_import_does_not_import_ml_stack() -> None:
    """Importing the MCP server should not import local embedding libraries."""
    code = """
import json
import sys

import librarian.server  # noqa: F401

print(json.dumps({
    name: name in sys.modules
    for name in ("torch", "transformers", "sentence_transformers")
}))
"""
    result = subprocess.run(  # noqa: S603 - controlled interpreter subprocess for import isolation
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    imported = json.loads(result.stdout)

    assert imported == {
        "torch": False,
        "transformers": False,
        "sentence_transformers": False,
    }

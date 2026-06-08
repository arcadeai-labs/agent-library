"""Tests for configuration validation."""

import os
import subprocess
import sys


def test_invalid_storage_backend_fails_at_config_load() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import librarian.config"],
        check=False,
        env={**os.environ, "STORAGE_BACKEND": "postgress"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Invalid STORAGE_BACKEND 'postgress'" in result.stderr

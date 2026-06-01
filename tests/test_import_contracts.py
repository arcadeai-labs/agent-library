"""Enforce the module-boundary contracts (import-linter) as part of the suite.

This keeps the private/public boundary from silently regressing: connectors must
stay storage-free, and the storage protocols must not directly depend on a
concrete backend. Skipped if import-linter isn't installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LINT_IMPORTS = shutil.which("lint-imports")

pytestmark = pytest.mark.skipif(
    _LINT_IMPORTS is None,
    reason="import-linter (lint-imports) not installed",
)


def test_import_contracts_hold() -> None:
    assert _LINT_IMPORTS is not None
    result = subprocess.run(  # noqa: S603 - fixed, trusted executable path
        [_LINT_IMPORTS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "import-linter contracts are broken:\n" + result.stdout + result.stderr
    )

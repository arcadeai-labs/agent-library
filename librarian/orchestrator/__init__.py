"""
Orchestrator -- the v0.14 ingest engine.

Consumes a connector's ``ChangeEvent`` stream and writes content + cursor
advances atomically into a ``Storage`` backend.

Usage:
    from librarian.connectors import LocalFileConnector
    from librarian.orchestrator import Orchestrator

    result = await Orchestrator().sync(LocalFileConnector(["~/notes"]))
"""

from librarian.orchestrator.orchestrator import Orchestrator, SyncResult

__all__ = ["Orchestrator", "SyncResult"]

"""
Connectors -- source adapters for the v0.14 single forward path.

A connector turns an external source into an async stream of ``ChangeEvent``s.
The built-in :class:`LocalFileConnector` powers the existing file-indexing
workflow (``libr add``) on top of this infrastructure.

Usage:
    from librarian.connectors import LocalFileConnector
    from librarian.orchestrator import Orchestrator

    connector = LocalFileConnector(["~/notes"])
    await Orchestrator().sync(connector)
"""

from librarian.connectors.base import (
    ChangeEvent,
    ChunkInput,
    Connector,
    DocumentSoftDelete,
    DocumentUpsert,
)
from librarian.connectors.local_file import LocalFileConnector

__all__ = [
    "ChangeEvent",
    "ChunkInput",
    "Connector",
    "DocumentSoftDelete",
    "DocumentUpsert",
    "LocalFileConnector",
]

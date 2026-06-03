"""
Orchestrator -- the single forward path's engine.

The orchestrator consumes a :class:`~librarian.connectors.Connector`'s
``ChangeEvent`` stream and, for each event, does the slow work (parsing,
chunking, embedding) *outside* a transaction, then writes the resulting content
and advances the connector's cursor *inside one* transaction. That single-
transaction guarantee is what makes a crash mid-stream resumable: either an
event's chunks and its cursor advance both commit, or neither does -- so there
are never orphan ``chunks`` rows or a cursor pointing past uncommitted content.

It owns the parse/chunk/embed routing that used to live in ``IndexingService``;
in v0.14 ``IndexingService`` becomes a thin shim over this class.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from librarian import ids
from librarian.config import ENABLE_CODE_EMBEDDINGS, ENABLE_VISION_EMBEDDINGS
from librarian.connectors.base import (
    Connector,
    DocumentSoftDelete,
    DocumentUpsert,
)
from librarian.connectors.local_file import LocalFileConnector
from librarian.processing.embed import get_embedder, get_embedder_for_modality
from librarian.processing.parsers.base import FileReadError, FileReadTimeoutError
from librarian.processing.parsers.registry import get_parser_for_file
from librarian.processing.transform.chunker import Chunker, ChunkingStrategy
from librarian.processing.transform.code import CodeChunker, chunk_code_by_blocks
from librarian.processing.transform.pdf import PDFChunker
from librarian.storage.factory import get_storage
from librarian.storage.protocols import SyncState
from librarian.storage.write_models import PreparedChunk, PreparedDocument
from librarian.types import AssetType, EmbeddingModality, ParsedDocument, TextChunk

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from librarian.storage.protocols import Storage

logger = logging.getLogger(__name__)

__all__ = ["Orchestrator", "SyncResult"]


@dataclass
class SyncResult:
    """Aggregate outcome of an :meth:`Orchestrator.sync` run."""

    source_key: str
    documents_upserted: int = 0
    documents_deleted: int = 0
    chunks_written: int = 0
    cursor: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Drives a connector's change stream into storage atomically."""

    def __init__(
        self,
        storage: "Storage | None" = None,
        embedder: Any = None,
    ) -> None:
        self.storage = storage or get_storage()
        self._embedder = embedder
        self._text_chunker = Chunker(strategy=ChunkingStrategy.HEADERS)
        self._code_chunker = CodeChunker()
        self._pdf_chunker = PDFChunker()

    async def sync(
        self,
        connector: Connector,
        *,
        source_key: str | None = None,
    ) -> SyncResult:
        """Pull changes from ``connector`` and persist them atomically.

        Args:
            connector: The source adapter to sync.
            source_key: The sync-state key. Defaults to the connector name.

        Returns:
            A :class:`SyncResult` with counts and the final cursor.
        """
        source_key = source_key or connector.name
        existing = self.storage.get_sync_state(source_key)
        cursor: dict[str, Any] = existing.cursor if existing else connector.initial_state()
        documents_seen = existing.documents_seen if existing else 0
        chunks_total = existing.chunks_written if existing else 0

        result = SyncResult(source_key=source_key, cursor=cursor)

        async for event in connector.fetch_changes(cursor):
            if isinstance(event, DocumentUpsert):
                prepared = self._prepare_upsert(connector, event)
                next_cursor = event.checkpoint if event.checkpoint is not None else cursor
                documents_seen += 1
                chunks_total += len(prepared.chunks)
                with self.storage.transaction() as conn:
                    self.storage.write_upsert(conn, prepared)
                    self.storage.put_sync_state(
                        SyncState(
                            source_key=source_key,
                            cursor=next_cursor,
                            status="ok",
                            last_success_at=datetime.now(),
                            last_attempt_at=datetime.now(),
                            documents_seen=documents_seen,
                            chunks_written=chunks_total,
                            config_version=existing.config_version if existing else 0,
                        ),
                        conn=conn,
                    )
                cursor = next_cursor
                result.documents_upserted += 1
                result.chunks_written += len(prepared.chunks)
            elif isinstance(event, DocumentSoftDelete):
                doc_id = ids.document_id(connector.name, event.source_type, event.source_native_id)
                next_cursor = event.checkpoint if event.checkpoint is not None else cursor
                with self.storage.transaction() as conn:
                    self.storage.soft_delete_document(conn, doc_id, event.deletion_reason)
                    self.storage.put_sync_state(
                        SyncState(
                            source_key=source_key,
                            cursor=next_cursor,
                            status="ok",
                            last_success_at=datetime.now(),
                            last_attempt_at=datetime.now(),
                            documents_seen=documents_seen,
                            chunks_written=chunks_total,
                            config_version=existing.config_version if existing else 0,
                        ),
                        conn=conn,
                    )
                cursor = next_cursor
                result.documents_deleted += 1

        result.cursor = cursor
        return result

    # =========================================================================
    # File-mode shim (synchronous single-file convenience)
    # =========================================================================

    def index_file(
        self, file_path: Path | str, *, source_key: str = "local_file"
    ) -> dict[str, Any]:
        """Index a single file synchronously (the file-mode driver).

        This is the synchronous entrypoint the ``IndexingService`` shim and the
        CLI/MCP file-ingest path use. It reproduces the old
        ``IndexingService.index_file`` surface (``{path, title, chunks, status}``
        plus a ``reason`` when skipped) while writing through the v0.14 storage
        path: deterministic ids, new columns, and a merged ``local_file`` cursor
        in ``sync_state``.

        Raises:
            FileReadTimeoutError: If ``stat()`` times out (cloud storage not synced).
            FileReadError: For other I/O errors accessing the file.
        """
        file_path = Path(file_path)
        try:
            mtime = file_path.stat().st_mtime
        except TimeoutError as e:
            raise FileReadTimeoutError(
                f"Timed out accessing {file_path} (file may not be synced from cloud storage)"
            ) from e
        except OSError as e:
            raise FileReadError(f"Cannot access {file_path}: {e}") from e

        connector = LocalFileConnector([file_path])
        event = connector.build_upsert(file_path, mtime)
        if event is None:
            return {
                "path": str(file_path),
                "title": None,
                "chunks": 0,
                "status": "skipped",
                "reason": "no parser found",
            }

        existing = self.storage.metadata.get_document_by_path(str(file_path))
        status = "updated" if existing else "created"
        prepared = self._prepare_upsert(connector, event)

        with self.storage.transaction() as conn:
            self.storage.write_upsert(conn, prepared)
            state = self.storage.get_sync_state(source_key) or SyncState(
                source_key=source_key, cursor={"mtimes": {}}
            )
            state.cursor.setdefault("mtimes", {})[str(file_path)] = mtime
            state.status = "ok"
            state.last_success_at = datetime.now()
            state.last_attempt_at = datetime.now()
            state.documents_seen += 1
            state.chunks_written += len(prepared.chunks)
            self.storage.put_sync_state(state, conn=conn)

        return {
            "path": str(file_path),
            "title": prepared.title,
            "chunks": len(prepared.chunks),
            "status": status,
        }

    # =========================================================================
    # Prepare (slow work, no DB writes)
    # =========================================================================

    def _prepare_upsert(self, connector: Connector, event: DocumentUpsert) -> PreparedDocument:
        document_id = ids.document_id(connector.name, event.source_type, event.source_native_id)

        if event.chunks is not None:
            text_chunks, asset_type, native_ids = self._chunks_from_inputs(event)
            content = "\n\n".join(c.content for c in text_chunks)
            title = event.title
        else:
            parsed, asset_type = self._parse(event)
            text_chunks = self._chunk_parsed(parsed, asset_type)
            native_ids = [f"{event.source_native_id}#chunk={i}" for i in range(len(text_chunks))]
            content = parsed.content
            title = event.title if event.title is not None else parsed.title

        modality, embeddings, model_version = self._embed(event, text_chunks, asset_type)

        prepared_chunks = [
            PreparedChunk(
                chunk_id=ids.chunk_id(connector.name, event.source_type, native_ids[i]),
                content=chunk.content,
                chunk_index=i,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                heading_path=chunk.heading_path,
                chunk_source_uri=(
                    (chunk.metadata or {}).get("chunk_source_uri")
                    or (
                        f"{event.document_source_uri}#chunk={i}"
                        if event.document_source_uri
                        else None
                    )
                ),
                asset_type=asset_type,
                modality=modality,
                embedding=embedding,
                model_version=model_version,
            )
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings, strict=True))
        ]

        return PreparedDocument(
            document_id=document_id,
            path=event.source_native_id,
            title=title,
            content=content,
            metadata=dict(event.metadata),
            asset_type=asset_type,
            document_source_uri=event.document_source_uri,
            source_created_at=event.source_created_at,
            document_size=len(content),
            file_mtime=event.metadata.get("file_mtime"),
            chunks=prepared_chunks,
        )

    def _chunks_from_inputs(
        self, event: DocumentUpsert
    ) -> tuple[list[TextChunk], AssetType, list[str]]:
        if not event.chunks:
            return [], event.asset_type, []
        text_chunks: list[TextChunk] = []
        native_ids: list[str] = []
        offset = 0
        for ci in event.chunks:
            text_chunks.append(
                TextChunk(
                    content=ci.content,
                    index=ci.chunk_index,
                    start_char=offset,
                    end_char=offset + len(ci.content),
                    heading_path=ci.heading_path,
                    metadata={**ci.metadata, "chunk_source_uri": ci.chunk_source_uri},
                )
            )
            native_ids.append(ci.source_native_id)
            offset += len(ci.content)
        asset_type = event.chunks[0].asset_type if event.chunks else event.asset_type
        return text_chunks, asset_type, native_ids

    def _parse(self, event: DocumentUpsert) -> tuple[ParsedDocument, AssetType]:
        path = Path(event.source_native_id)
        parser, asset_type = get_parser_for_file(path)
        if parser is None:
            # Unknown type: treat the raw text (if any) as a single plain document.
            raw = event.raw_content if isinstance(event.raw_content, str) else ""
            parsed = ParsedDocument(
                path=str(path),
                title=event.title or path.name,
                content=raw,
                metadata=dict(event.metadata),
                sections=[],
                raw_content=raw,
                asset_type=AssetType.TEXT,
            )
            return parsed, AssetType.TEXT

        if isinstance(event.raw_content, str):
            parsed = parser.parse_content(event.raw_content, str(path))
        else:
            parsed = parser.parse_file(path)
        return parsed, asset_type

    def _chunk_parsed(self, parsed: ParsedDocument, asset_type: AssetType) -> list[TextChunk]:
        if asset_type == AssetType.CODE:
            symbols = parsed.metadata.get("symbols", [])
            if symbols:
                from librarian.types import CodeSymbol, CodeSymbolType

                code_symbols = [
                    CodeSymbol(
                        name=s["name"],
                        symbol_type=CodeSymbolType(s["type"]),
                        line_start=s["line_start"],
                        line_end=s["line_end"],
                    )
                    for s in symbols
                ]
                return self._code_chunker.chunk_by_symbols(
                    parsed.content, code_symbols, parsed.metadata
                )
            language = parsed.metadata.get("language", "unknown")
            return chunk_code_by_blocks(parsed.content, language, parsed.metadata)
        if asset_type == AssetType.PDF:
            page_count = parsed.metadata.get("page_count", 1)
            return self._pdf_chunker.chunk_by_pages(parsed.content, page_count, parsed.metadata)
        if asset_type == AssetType.IMAGE:
            return [
                TextChunk(
                    content=parsed.content,
                    index=0,
                    start_char=0,
                    end_char=len(parsed.content),
                    heading_path=parsed.title,
                    metadata=parsed.metadata,
                )
            ]
        return self._text_chunker.chunk_document(parsed)

    # =========================================================================
    # Embedding (ported from IndexingService)
    # =========================================================================

    def _determine_modality(self, asset_type: AssetType) -> EmbeddingModality:
        if asset_type == AssetType.CODE and ENABLE_CODE_EMBEDDINGS:
            return EmbeddingModality.CODE
        if asset_type == AssetType.IMAGE and ENABLE_VISION_EMBEDDINGS:
            return EmbeddingModality.VISION
        return EmbeddingModality.TEXT

    def _embed(
        self,
        event: DocumentUpsert,
        chunks: list[TextChunk],
        asset_type: AssetType,
    ) -> tuple[EmbeddingModality, list[list[float]], str]:
        modality = self._determine_modality(asset_type)
        embedder = self._embedder or get_embedder_for_modality(modality)
        if embedder is None:
            modality = EmbeddingModality.TEXT
            embedder = self._embedder or get_embedder()

        if (
            modality == EmbeddingModality.VISION
            and asset_type == AssetType.IMAGE
            and not isinstance(event.raw_content, str)
        ):
            embeddings = self._embed_image_chunks(Path(event.source_native_id), chunks, embedder)
        else:
            embeddings = embedder.embed_documents([c.content for c in chunks])

        model_version = getattr(embedder, "model_name", "unknown")
        return modality, embeddings, model_version

    def _load_image(self, file_path: Path) -> "PILImage | None":  # type: ignore[no-any-unimported]
        try:
            from PIL import Image

            return Image.open(file_path)
        except ImportError:
            logger.warning("PIL not available for image loading")
            return None
        except Exception as e:
            logger.warning("Failed to load image %s: %s", file_path, e)
            return None

    def _embed_image_chunks(
        self, file_path: Path, chunks: list[TextChunk], embedder: Any
    ) -> list[list[float]]:
        img = self._load_image(file_path)
        if img is None:
            fallback: list[list[float]] = embedder.embed_documents([c.content for c in chunks])
            return fallback
        try:
            if hasattr(embedder, "embed_image"):
                embedding = embedder.embed_image(img)
            else:
                embedding = embedder.embed(img)
            img.close()
            return [embedding] * len(chunks)
        except Exception as e:
            logger.warning("Vision embedding failed for %s: %s; text fallback", file_path, e)
            img.close()
            result: list[list[float]] = embedder.embed_documents([c.content for c in chunks])
            return result

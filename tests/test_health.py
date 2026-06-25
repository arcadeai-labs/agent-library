"""Tests for library health diagnostics."""

from pathlib import Path

from librarian.health import run_health_check
from librarian.storage.database import Database
from librarian.types import AssetType, Chunk, Document, EmbeddingModality


def test_health_check_reports_retrieval_risk_signals(tmp_path: Path) -> None:
    db = Database(db_path=str(tmp_path / "health.db"))

    note_path = tmp_path / "note.md"
    note_path.write_text("# Stable Note\n\nUseful content about agents.", encoding="utf-8")
    note_doc_id = db.insert_document(
        Document(
            id=None,
            path=str(note_path),
            title="Stable Note",
            content=note_path.read_text(encoding="utf-8"),
            metadata={},
            file_mtime=note_path.stat().st_mtime,
            asset_type=AssetType.TEXT,
        )
    )
    db.insert_chunk(
        Chunk(
            id=None,
            document_id=note_doc_id,
            content="Useful content about agents.",
            heading_path="Stable Note",
            chunk_index=0,
            start_char=0,
            end_char=28,
            embedding=[0.1] * 384,
            asset_type=AssetType.TEXT,
            modality=EmbeddingModality.TEXT,
        )
    )

    pdf_doc_id = db.insert_document(
        Document(
            id=None,
            path=str(tmp_path / "scanned.pdf"),
            title="Scanned",
            content="tiny",
            metadata={"page_count": 12},
            file_mtime=0.0,
            asset_type=AssetType.PDF,
        )
    )
    db.insert_chunk(
        Chunk(
            id=None,
            document_id=pdf_doc_id,
            content="tiny",
            heading_path="Page 1",
            chunk_index=0,
            start_char=0,
            end_char=4,
            embedding=None,
            asset_type=AssetType.PDF,
            modality=EmbeddingModality.TEXT,
        )
    )

    image_doc_id = db.insert_document(
        Document(
            id=None,
            path=str(tmp_path / "screenshot.png"),
            title="Screenshot",
            content="Image: screenshot.png\nFormat: PNG",
            metadata={"format": "PNG", "width": 100, "height": 100},
            file_mtime=0.0,
            asset_type=AssetType.IMAGE,
        )
    )
    db.insert_chunk(
        Chunk(
            id=None,
            document_id=image_doc_id,
            content="Image: screenshot.png\nFormat: PNG",
            heading_path="Screenshot",
            chunk_index=0,
            start_char=0,
            end_char=32,
            embedding=None,
            asset_type=AssetType.IMAGE,
            modality=EmbeddingModality.VISION,
        )
    )

    report = run_health_check(db=db, sources=[], sample_limit=10)
    codes = {issue.code for issue in report.issues}

    assert report.document_count == 3
    assert report.chunk_count == 3
    assert report.embedding_counts["text"] == 1
    assert report.embedding_count == 1
    assert report.embedding_coverage == 0.3333
    assert report.document_asset_counts == {"image": 1, "pdf": 1, "text": 1}
    assert "missing_embeddings" in codes
    assert "pdf_text_too_short" in codes
    assert "image_no_ocr_text" in codes
    assert "small_chunks" in codes


def test_health_check_reports_missing_sources(tmp_path: Path) -> None:
    db = Database(db_path=str(tmp_path / "health.db"))
    missing_source = tmp_path / "missing"

    report = run_health_check(
        db=db,
        sources=[{"name": "missing", "path": str(missing_source)}],
        sample_limit=10,
    )

    codes = {issue.code for issue in report.issues}
    assert "empty_index" in codes
    assert "source_path_missing" in codes

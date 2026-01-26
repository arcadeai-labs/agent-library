"""Tests for OCR functionality in ImageParser."""

from pathlib import Path

import pytest

from librarian.processing.parsers.image import ImageParser
from librarian.types import AssetType


class TestOCRFunctionality:
    """Test OCR text extraction from images."""

    def test_image_parser_without_ocr(self) -> None:
        """Test image parsing with OCR disabled."""
        try:
            parser = ImageParser(enable_ocr=False)
        except ImportError:
            pytest.skip("PIL not installed")
        test_file = Path(__file__).parent / "data" / "ocr_test_screenshot.png"

        if not test_file.exists():
            pytest.skip("Test image not found")

        parsed = parser.parse_file(test_file)

        assert parsed.asset_type == AssetType.IMAGE
        assert parsed.title == "ocr_test_screenshot"
        assert "ocr_text" not in parsed.metadata
        assert "Text extracted from image" not in parsed.content

    def test_image_parser_with_ocr(self) -> None:
        """Test image parsing with OCR enabled."""
        try:
            import pytesseract

            # Check if tesseract is available
            pytesseract.get_tesseract_version()
            parser = ImageParser(enable_ocr=True)
        except ImportError:
            pytest.skip("PIL or pytesseract not installed")
        except Exception:
            pytest.skip("Tesseract not installed or not in PATH")
        test_file = Path(__file__).parent / "data" / "ocr_test_screenshot.png"

        if not test_file.exists():
            pytest.skip("Test image not found")

        parsed = parser.parse_file(test_file)

        assert parsed.asset_type == AssetType.IMAGE
        assert parsed.title == "ocr_test_screenshot"
        assert "ocr_text" in parsed.metadata
        assert "Text extracted from image" in parsed.content

        # Check that OCR extracted some text
        ocr_text = parsed.metadata["ocr_text"]
        assert len(ocr_text) > 0

        # Check for expected text (case-insensitive)
        ocr_lower = ocr_text.lower()
        assert "todo" in ocr_lower or "authentication" in ocr_lower

    def test_ocr_with_blank_image(self) -> None:
        """Test OCR with an image containing no text."""
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except Exception:
            pytest.skip("Tesseract not installed")

        # Use the test_diagram.png which has minimal text
        parser = ImageParser(enable_ocr=True)
        test_file = Path(__file__).parent / "data" / "test_diagram.png"

        if not test_file.exists():
            pytest.skip("Test image not found")

        parsed = parser.parse_file(test_file)

        # Should still work, just with little/no extracted text
        assert parsed.asset_type == AssetType.IMAGE
        assert "Image Information" in parsed.sections[0].title


class TestOCRIntegration:
    """Test OCR integration with indexing and search."""

    def test_ocr_searchability(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """Test that OCR-extracted text is searchable."""
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except Exception:
            pytest.skip("Tesseract not installed")

        # Index image with OCR enabled
        import os

        from librarian.indexing import get_indexing_service
        from librarian.storage.database import get_database

        os.environ["ENABLE_OCR"] = "true"

        service = get_indexing_service()
        test_file = Path(__file__).parent / "data" / "ocr_test_screenshot.png"

        if not test_file.exists():
            pytest.skip("Test image not found")

        result = service.index_file(test_file)

        assert result["status"] in ("created", "updated")
        assert result["chunks"] > 0

        # Verify OCR text was stored in database
        db = get_database()
        doc = db.get_document_by_path(str(test_file))
        assert doc is not None
        assert "TODO" in doc.content or "authentication" in doc.content, (
            "OCR text should be in document content"
        )

        # Verify chunks contain OCR text
        chunks = db.get_chunks_by_document(doc.id)
        assert len(chunks) > 0
        chunk_text = " ".join([c.content for c in chunks])
        assert "TODO" in chunk_text or "authentication" in chunk_text, (
            "OCR text should be in chunks"
        )

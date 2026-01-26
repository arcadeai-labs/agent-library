"""
Image parser for extracting metadata and descriptions from images.

Uses PIL for image metadata and EXIF data extraction.
Optionally uses pytesseract for OCR text extraction.
"""

import logging
from pathlib import Path
from typing import Any

from librarian.config import ENABLE_OCR, OCR_CONFIG, OCR_LANGUAGE, OCR_MIN_CONFIDENCE
from librarian.processing.parsers.base import BaseParser
from librarian.types import AssetType, ParsedDocument, Section

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    from PIL.Image import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PILImage = Any  # type: ignore[misc,assignment]

try:
    import pytesseract

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class ImageParser(BaseParser):
    """Parser for image files with optional OCR support."""

    def __init__(self, enable_ocr: bool | None = None) -> None:
        """
        Initialize image parser.

        Args:
            enable_ocr: Enable OCR text extraction. Defaults to ENABLE_OCR config.
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for image parsing. Install with: pip install Pillow"
            )

        self.enable_ocr = enable_ocr if enable_ocr is not None else ENABLE_OCR

        if self.enable_ocr and not PYTESSERACT_AVAILABLE:
            logger.warning(
                "OCR enabled but pytesseract not installed. Install with: pip install pytesseract"
            )
            self.enable_ocr = False

    def parse_content(self, content: str, path: str = "") -> ParsedDocument:
        """
        Parse image content from a string.

        Note: Images require binary data and cannot be parsed from string content.
        This method is provided to satisfy the BaseParser interface.

        Args:
            content: The content string (not supported for images).
            path: Optional path for reference.

        Raises:
            NotImplementedError: Images must be parsed from file path.
        """
        raise NotImplementedError(
            "Image parser requires a file path. Use parse_file() instead of parse_content()."
        )

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse an image file.

        Args:
            file_path: Path to the image file.

        Returns:
            ParsedDocument with image metadata and EXIF data.
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Open image
        with Image.open(file_path) as img:
            # Extract basic metadata
            metadata: dict[str, Any] = {
                "file_type": "image",
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_bytes": file_path.stat().st_size,
            }

            # Extract EXIF data if available
            exif_data = self._extract_exif(img)
            if exif_data:
                metadata["exif"] = exif_data

            # Create content description
            content_parts = [
                f"Image: {file_path.name}",
                f"Format: {img.format}",
                f"Dimensions: {img.width}x{img.height}",
                f"Mode: {img.mode}",
            ]

            # Add EXIF highlights to content
            if exif_data:
                if "DateTime" in exif_data:
                    content_parts.append(f"Captured: {exif_data['DateTime']}")
                if "Make" in exif_data and "Model" in exif_data:
                    content_parts.append(f"Camera: {exif_data['Make']} {exif_data['Model']}")
                if "GPSInfo" in exif_data:
                    content_parts.append("Location: GPS data available")

            # Extract text via OCR if enabled
            ocr_text = ""
            if self.enable_ocr:
                ocr_text = self._extract_ocr_text(img)
                if ocr_text:
                    metadata["ocr_text"] = ocr_text

            content = "\n".join(content_parts)

            # Add OCR text to content if available
            if ocr_text:
                content += f"\n\nText extracted from image:\n{ocr_text}"

            # Create sections
            sections = []

            # Basic info section
            sections.append(
                Section(
                    title="Image Information",
                    level=1,
                    content=content,
                    start_pos=0,
                    end_pos=len(content),
                )
            )

            # EXIF section if available
            if exif_data:
                exif_content = "\n".join([f"{k}: {v}" for k, v in exif_data.items()])
                sections.append(
                    Section(
                        title="EXIF Data",
                        level=1,
                        content=exif_content,
                        start_pos=len(content),
                        end_pos=len(content) + len(exif_content),
                    )
                )

            # Use filename as title
            title = file_path.stem

            return ParsedDocument(
                path=str(file_path),
                title=title,
                content=content,
                metadata=metadata,
                sections=sections,
                raw_content=content,
                asset_type=AssetType.IMAGE,
                modality_data={
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                },
            )

    def _extract_exif(self, img: PILImage) -> dict[str, Any]:  # type: ignore[misc,no-any-unimported]
        """
        Extract EXIF data from an image.

        Args:
            img: PIL Image object.

        Returns:
            Dictionary of EXIF data.
        """
        exif_data: dict[str, Any] = {}

        try:
            exif = img.getexif()
            if exif is None:
                return exif_data

            for tag_id, value in exif.items():
                # Get tag name
                tag = TAGS.get(tag_id, tag_id)

                # Convert value to JSON-serializable format
                if isinstance(value, bytes):
                    # Skip binary data
                    continue
                elif isinstance(value, tuple):
                    value = list(value)
                elif hasattr(value, "__class__") and "IFD" in value.__class__.__name__:
                    # Handle PIL IFDRational and similar types
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = str(value)

                exif_data[str(tag)] = value

        except Exception as e:
            logger.warning(f"Error extracting EXIF data: {e}")

        return exif_data

    def _extract_ocr_text(self, img: PILImage) -> str:  # type: ignore[misc,no-any-unimported]
        """
        Extract text from an image using OCR.

        Args:
            img: PIL Image object.

        Returns:
            Extracted text string, or empty string if no text found or error.
        """
        if not PYTESSERACT_AVAILABLE:
            return ""

        try:
            # Configure pytesseract
            config = OCR_CONFIG
            if OCR_LANGUAGE != "eng":
                config = f"-l {OCR_LANGUAGE} {config}"

            # Extract text
            text: str = str(pytesseract.image_to_string(img, config=config))

            # Clean up text
            text = text.strip()

            # Filter by confidence if needed
            if OCR_MIN_CONFIDENCE > 0:
                # Get detailed data with confidence scores
                data = pytesseract.image_to_data(
                    img, config=config, output_type=pytesseract.Output.DICT
                )  # type: ignore[attr-defined]
                filtered_text_parts = []

                for i, conf in enumerate(data["conf"]):  # type: ignore[index]
                    if conf != -1 and conf >= OCR_MIN_CONFIDENCE:
                        filtered_text_parts.append(data["text"][i])  # type: ignore[index]

                return " ".join(filtered_text_parts).strip()
            else:
                return text

        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""

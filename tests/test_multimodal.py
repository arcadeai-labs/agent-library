"""Integration tests for multi-modal parsing and indexing."""

import contextlib
from pathlib import Path

import pytest

from librarian.processing.parsers.code import CodeParser
from librarian.processing.parsers.registry import get_parser_for_file
from librarian.types import AssetType, CodeSymbolType, ProgrammingLanguage


class TestCodeParser:
    """Test code parser with real Python files."""

    def test_parse_python_file(self) -> None:
        """Test parsing a real Python file."""
        parser = CodeParser()
        test_file = Path(__file__).parent / "data" / "code" / "example.py"

        parsed = parser.parse_file(test_file)

        assert parsed.asset_type == AssetType.CODE
        assert parsed.metadata["language"] == "python"
        assert parsed.title == "example"
        assert len(parsed.sections) > 0

        # Check symbols were extracted
        symbols = parsed.metadata.get("symbols", [])
        assert len(symbols) > 0

        # Should find Calculator class
        class_symbols = [s for s in symbols if s["type"] == "class"]
        assert len(class_symbols) >= 1
        assert any(s["name"] == "Calculator" for s in class_symbols)

        # Should find methods
        method_symbols = [s for s in symbols if s["type"] == "method"]
        assert len(method_symbols) >= 2  # __init__, add, multiply

        # Should find functions
        func_symbols = [s for s in symbols if s["type"] == "function"]
        assert len(func_symbols) >= 2  # create_calculator, async_calculate

    def test_parse_javascript_content(self) -> None:
        """Test parsing JavaScript code."""
        parser = CodeParser(language=ProgrammingLanguage.JAVASCRIPT)
        js_code = """
function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name) {
        this.name = name;
    }

    sayHello() {
        return greet(this.name);
    }
}

const createPerson = (name) => new Person(name);
"""
        # Create temp file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            temp_path = Path(f.name)

        try:
            parsed = parser.parse_file(temp_path)
            assert parsed.asset_type == AssetType.CODE
            assert parsed.metadata["language"] == "javascript"

            symbols = parsed.metadata.get("symbols", [])
            # Should find class and functions
            assert len(symbols) >= 2
        finally:
            temp_path.unlink()


class TestParserRegistry:
    """Test parser registry with different file types."""

    def test_registry_selects_code_parser(self) -> None:
        """Test registry selects code parser for .py files."""
        test_file = Path(__file__).parent / "data" / "code" / "example.py"
        parser, asset_type = get_parser_for_file(test_file)

        assert asset_type == AssetType.CODE
        assert parser is not None
        assert isinstance(parser, CodeParser)

    def test_registry_detects_languages(self) -> None:
        """Test registry detects different programming languages."""
        from librarian.processing.parsers.registry import get_registry

        registry = get_registry()

        assert registry.get_language_for_extension(".py") == ProgrammingLanguage.PYTHON
        assert registry.get_language_for_extension(".js") == ProgrammingLanguage.JAVASCRIPT
        assert registry.get_language_for_extension(".go") == ProgrammingLanguage.GO
        assert registry.get_language_for_extension(".rs") == ProgrammingLanguage.RUST

    def test_registry_asset_types(self) -> None:
        """Test registry correctly identifies asset types."""
        from librarian.processing.parsers.registry import get_registry

        registry = get_registry()

        assert registry.get_asset_type(Path("file.md")) == AssetType.TEXT
        assert registry.get_asset_type(Path("file.py")) == AssetType.CODE
        assert registry.get_asset_type(Path("file.js")) == AssetType.CODE

    def test_image_parser_uses_configured_ocr_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Image parser should not silently disable OCR when ENABLE_OCR is unset."""
        from librarian import config as config_module
        from librarian.processing.parsers.image import ImageParser

        monkeypatch.delenv("ENABLE_OCR", raising=False)
        monkeypatch.setattr(config_module, "ENABLE_OCR", True)

        test_file = Path(__file__).parent / "data" / "test_diagram.png"
        parser, asset_type = get_parser_for_file(test_file)

        assert asset_type == AssetType.IMAGE
        assert isinstance(parser, ImageParser)
        assert parser.enable_ocr is True


class TestCodeChunking:
    """Test code-specific chunking."""

    def test_chunk_by_symbols(self) -> None:
        """Test chunking code by symbols."""
        from librarian.processing.transform.code import CodeChunker
        from librarian.types import CodeSymbol

        chunker = CodeChunker()
        code = """def foo():
    pass

def bar():
    return 42

class Baz:
    def method(self):
        pass
"""
        symbols = [
            CodeSymbol("foo", CodeSymbolType.FUNCTION, 1, 2),
            CodeSymbol("bar", CodeSymbolType.FUNCTION, 4, 5),
            CodeSymbol("Baz", CodeSymbolType.CLASS, 7, 9),
        ]

        chunks = chunker.chunk_by_symbols(code, symbols)

        assert len(chunks) == 3
        assert "foo" in chunks[0].content
        assert "bar" in chunks[1].content
        assert "Baz" in chunks[2].content

    def test_chunk_code_blocks(self) -> None:
        """Test block-based code chunking."""
        from librarian.processing.transform.code import chunk_code_by_blocks

        code = """def function1():
    return 1

def function2():
    return 2

class MyClass:
    pass
"""
        chunks = chunk_code_by_blocks(code, "python")

        assert len(chunks) >= 3  # Should find function and class boundaries


class TestPDFParser:
    """Test PDF parser with real PDF file."""

    def test_pdf_parser_availability(self) -> None:
        """Test if PDF parser can be imported."""
        try:
            from librarian.processing.parsers.pdf import PDFParser

            parser = PDFParser()
            assert parser is not None
        except ImportError:
            pytest.skip("PDF dependencies not installed")

    def test_parse_pdf_file(self) -> None:
        """Test parsing a real PDF file."""
        try:
            from librarian.processing.parsers.pdf import PDFParser

            parser = PDFParser()
        except ImportError:
            pytest.skip("PDF dependencies not installed")
        test_file = Path(__file__).parent / "data" / "test.pdf"

        if not test_file.exists():
            pytest.skip("Test PDF file not found")

        parsed = parser.parse_file(test_file)

        assert parsed.asset_type == AssetType.PDF
        assert parsed.metadata["file_type"] == "pdf"
        assert parsed.metadata["page_count"] == 2
        assert len(parsed.content) > 0

        # Should have sections for each page
        assert len(parsed.sections) == 2
        assert "Test PDF Document" in parsed.content
        assert "Page 2: Technical Details" in parsed.content


class TestImageParser:
    """Test image parser with real image file."""

    def test_image_parser_availability(self) -> None:
        """Test if image parser can be imported."""
        try:
            from librarian.processing.parsers.image import ImageParser

            parser = ImageParser()
            assert parser is not None
        except ImportError:
            pytest.skip("PIL not installed")

    def test_parse_image_file(self) -> None:
        """Test parsing a real image file."""
        try:
            from librarian.processing.parsers.image import ImageParser

            parser = ImageParser()
        except ImportError:
            pytest.skip("PIL not installed")
        test_file = Path(__file__).parent / "data" / "test_diagram.png"

        if not test_file.exists():
            pytest.skip("Test image file not found")

        parsed = parser.parse_file(test_file)

        assert parsed.asset_type == AssetType.IMAGE
        assert parsed.metadata["file_type"] == "image"
        assert parsed.metadata["format"] == "PNG"
        assert parsed.metadata["width"] == 400
        assert parsed.metadata["height"] == 300
        assert len(parsed.content) > 0

        # Should have image information section
        assert len(parsed.sections) >= 1
        assert "Image Information" in parsed.sections[0].title


class TestMultiModalSearch:
    """Test multi-modal search with correct asset type preservation."""

    def test_search_preserves_asset_types(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """Test that search results preserve correct asset types."""
        from librarian.indexing import get_indexing_service
        from librarian.retrieval.search import HybridSearcher
        from librarian.types import AssetType

        service = get_indexing_service()
        search = HybridSearcher(embedder=fake_embedder)

        # Index all three file types
        code_file = Path(__file__).parent / "data" / "code" / "example.py"
        pdf_file = Path(__file__).parent / "data" / "test.pdf"
        image_file = Path(__file__).parent / "data" / "test_diagram.png"

        if code_file.exists():
            service.index_file(code_file)
        if pdf_file.exists():
            with contextlib.suppress(ImportError):
                service.index_file(pdf_file)
        if image_file.exists():
            with contextlib.suppress(ImportError):
                service.index_file(image_file)

        # Search and verify asset types
        results = search.search("calculator", limit=10)

        # Should find code file
        code_results = [r for r in results if "example.py" in r.document_path]
        assert len(code_results) > 0
        for result in code_results:
            assert result.asset_type == AssetType.CODE

        # Search for PDF content
        results = search.search("PDF document", limit=10)
        pdf_results = [r for r in results if "test.pdf" in r.document_path]
        if len(pdf_results) > 0:
            for result in pdf_results:
                assert result.asset_type == AssetType.PDF

        # Search for image
        results = search.search("diagram", limit=10)
        image_results = [r for r in results if "test_diagram.png" in r.document_path]
        if len(image_results) > 0:
            for result in image_results:
                assert result.asset_type == AssetType.IMAGE

    def test_vector_search_asset_type(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """Test that vector-only search preserves asset type."""
        from librarian.indexing import get_indexing_service
        from librarian.retrieval.search import HybridSearcher
        from librarian.types import AssetType, EmbeddingModality

        service = get_indexing_service()
        search = HybridSearcher(embedder=fake_embedder)

        # Index code file
        code_file = Path(__file__).parent / "data" / "code" / "example.py"
        if code_file.exists():
            service.index_file(code_file)

            # Code embeddings are now stored in CODE modality table,
            # so use modality-specific search
            results = search.vector_search_by_modality(
                "calculator", EmbeddingModality.CODE, limit=5
            )

            # Fake embedder may return empty results, but if there are results,
            # they should have correct asset type
            for result in results:
                assert result.asset_type == AssetType.CODE

    def test_keyword_search_asset_type(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """Test that keyword-only search preserves asset type."""
        from librarian.indexing import get_indexing_service
        from librarian.retrieval.search import HybridSearcher
        from librarian.types import AssetType

        service = get_indexing_service()
        search = HybridSearcher(embedder=fake_embedder)

        # Index code file
        code_file = Path(__file__).parent / "data" / "code" / "example.py"
        if code_file.exists():
            service.index_file(code_file)

            # Keyword-only search
            results = search.keyword_search("calculator", limit=5)

            assert len(results) > 0
            for result in results:
                assert result.asset_type == AssetType.CODE


class TestMultiModalScoring:
    """Lock in the scoring contract for _merge_multi_modal_results.

    Regression: scores used to be divided by the total weight of all
    enabled modalities, pinning single-modality hits at 1/N (the observed
    0.25 ceiling when four modalities were active). Scores must now span
    a usable range with single-modality best hits near 1.0 and cross-modal
    agreement scored slightly above single-modality alternatives.
    """

    def _merge(self, search, modality_results):  # type: ignore[no-untyped-def]
        """Invoke _merge_multi_modal_results with synthetic per-modality results."""
        return search._merge_multi_modal_results(modality_results)

    def _mk(self, chunk_id, score, asset_type=AssetType.TEXT):  # type: ignore[no-untyped-def]
        from librarian.types import SearchResult

        return SearchResult(
            chunk_id=chunk_id,
            document_id=chunk_id,
            document_path=f"/doc/{chunk_id}",
            content="",
            heading_path=None,
            score=score,
            asset_type=asset_type,
        )

    def test_single_modality_hit_is_not_diluted(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """A chunk matched only in FTS should score ~1.0, not 1/num_modalities."""
        from librarian.retrieval.search import HybridSearcher

        search = HybridSearcher(embedder=fake_embedder)
        merged = self._merge(
            search,
            {
                "text": [self._mk(100, 0.5)],
                "code": [self._mk(200, 0.5)],
                "vision": [self._mk(300, 0.5)],
                "fts": [self._mk(1, 10.0)],  # raw score doesn't matter; normalized to 1.0
            },
        )
        winner = next(r for r in merged if r.chunk_id == 1)
        assert winner.score == pytest.approx(1.0, abs=1e-6)

    def test_cross_modal_agreement_ranks_above_single_match(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """A chunk matched in two modalities should outrank equally-scored single-match chunks."""
        from librarian.retrieval.search import HybridSearcher

        search = HybridSearcher(embedder=fake_embedder)
        merged = self._merge(
            search,
            {
                # Chunk 1: matches both text and FTS at their modality-best
                "text": [self._mk(1, 1.0), self._mk(2, 1.0)],
                "fts": [self._mk(1, 10.0), self._mk(3, 10.0)],
            },
        )
        by_id = {r.chunk_id: r.score for r in merged}
        # 1 appears in both modalities: 1.0 * (1 + 0.05) clamped to 1.0
        # 2 appears only in text: 1.0 * 1.0 = 1.0
        # 3 appears only in FTS: 1.0 * 1.0 = 1.0
        # All cap at 1.0 — the overlap bonus matters more at non-max scores.
        assert by_id[1] >= by_id[2]
        assert by_id[1] >= by_id[3]

    def test_overlap_bonus_lifts_multi_match_at_mid_scores(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """At non-saturating normalized scores, cross-modal agreement outranks single-match.

        Per-modality normalization divides each chunk's raw score by the
        modality's max, so to exercise the overlap bonus we need the
        matched-in-both chunk to be *below* its modality's max.
        """
        from librarian.retrieval.search import HybridSearcher

        search = HybridSearcher(embedder=fake_embedder)
        merged = self._merge(
            search,
            {
                # text max=10; chunk 1 → 0.8 normalized; chunk 2 → 0.8 normalized
                "text": [self._mk(1, 8.0), self._mk(2, 8.0), self._mk(0, 10.0)],
                # fts max=10; chunk 1 → 0.8 normalized; chunk 3 → 1.0 normalized
                "fts": [self._mk(1, 8.0), self._mk(3, 10.0)],
            },
        )
        by_id = {r.chunk_id: r.score for r in merged}
        # chunk 1 in both modalities @ 0.8 each: avg=0.8 * bonus=1.05 → 0.84
        # chunk 2 in text only @ 0.8: avg=0.8 * 1.0 → 0.80
        assert by_id[1] > by_id[2]
        assert by_id[1] == pytest.approx(0.84, abs=1e-3)
        assert by_id[2] == pytest.approx(0.80, abs=1e-3)

    def test_scores_stay_bounded_in_unit_interval(self, clean_db, fake_embedder) -> None:  # type: ignore[no-untyped-def]
        """No score should exceed 1.0 even with four-way cross-modal agreement."""
        from librarian.retrieval.search import HybridSearcher

        search = HybridSearcher(embedder=fake_embedder)
        merged = self._merge(
            search,
            {
                "text": [self._mk(1, 1.0)],
                "code": [self._mk(1, 1.0)],
                "vision": [self._mk(1, 1.0)],
                "fts": [self._mk(1, 10.0)],
            },
        )
        assert all(0.0 <= r.score <= 1.0 for r in merged)
        winner = next(r for r in merged if r.chunk_id == 1)
        assert winner.score == pytest.approx(1.0, abs=1e-6)

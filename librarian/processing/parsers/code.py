"""
Code parser for extracting structure from source files.

Supports multiple programming languages with regex-based parsing initially.
Can be extended with tree-sitter for more accurate AST parsing.
"""

import re
from pathlib import Path

from librarian.processing.parsers.base import BaseParser
from librarian.types import (
    AssetType,
    CodeSymbol,
    CodeSymbolType,
    ParsedDocument,
    ProgrammingLanguage,
    Section,
)


class CodeParser(BaseParser):
    """Parser for source code files."""

    def __init__(self, language: ProgrammingLanguage | None = None) -> None:
        """
        Initialize the code parser.

        Args:
            language: Programming language to parse. Auto-detected if None.
        """
        self.language = language
        self._language_map = {
            ".py": ProgrammingLanguage.PYTHON,
            ".js": ProgrammingLanguage.JAVASCRIPT,
            ".ts": ProgrammingLanguage.TYPESCRIPT,
            ".jsx": ProgrammingLanguage.JAVASCRIPT,
            ".tsx": ProgrammingLanguage.TYPESCRIPT,
            ".go": ProgrammingLanguage.GO,
            ".rs": ProgrammingLanguage.RUST,
            ".java": ProgrammingLanguage.JAVA,
            ".cpp": ProgrammingLanguage.CPP,
            ".cc": ProgrammingLanguage.CPP,
            ".c": ProgrammingLanguage.C,
            ".rb": ProgrammingLanguage.RUBY,
            ".php": ProgrammingLanguage.PHP,
            ".swift": ProgrammingLanguage.SWIFT,
            ".kt": ProgrammingLanguage.KOTLIN,
            ".cs": ProgrammingLanguage.CSHARP,
        }

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a source code file.

        Args:
            file_path: Path to the source file.

        Returns:
            ParsedDocument with extracted code structure.
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Read file
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Detect language if not specified
        language = self.language or self._detect_language(file_path)

        # Extract symbols (functions, classes, etc.)
        symbols = self._extract_symbols(content, language)

        # Create sections from symbols
        sections = self._symbols_to_sections(symbols, content)

        # Extract metadata
        metadata = {
            "language": language.value if language else "unknown",
            "file_type": "source_code",
            "extension": file_path.suffix,
            "symbols": [
                {
                    "name": s.name,
                    "type": s.symbol_type.value,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                }
                for s in symbols
            ],
        }

        # Use filename as title
        title = file_path.stem

        return ParsedDocument(
            path=str(file_path),
            title=title,
            content=content,
            metadata=metadata,
            sections=sections,
            raw_content=content,
            asset_type=AssetType.CODE,
            modality_data={"language": language.value if language else "unknown"},
        )

    def parse_content(self, content: str, path: str = "") -> ParsedDocument:
        """
        Parse source code content from a string.

        Args:
            content: Source code content.
            path: Optional path for reference.

        Returns:
            ParsedDocument with extracted code structure.
        """
        # Detect language from path extension if available
        language = self.language
        if not language and path:
            language = self._detect_language(Path(path))

        if not language:
            language = ProgrammingLanguage.OTHER

        # Extract symbols
        symbols = self._extract_symbols(content, language)
        sections = self._symbols_to_sections(symbols, content)

        # Extract metadata
        metadata = {
            "language": language.value if language else "unknown",
            "file_type": "source_code",
            "symbols": [
                {
                    "name": s.name,
                    "type": s.symbol_type.value,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                }
                for s in symbols
            ],
        }

        title = Path(path).stem if path else "code"

        return ParsedDocument(
            path=path,
            title=title,
            content=content,
            metadata=metadata,
            sections=sections,
            raw_content=content,
            asset_type=AssetType.CODE,
            modality_data={"language": language.value if language else "unknown"},
        )

    def _detect_language(self, file_path: Path) -> ProgrammingLanguage:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the source file.

        Returns:
            Programming language enum.
        """
        extension = file_path.suffix.lower()
        return self._language_map.get(extension, ProgrammingLanguage.OTHER)

    def _extract_symbols(
        self,
        content: str,
        language: ProgrammingLanguage,
    ) -> list[CodeSymbol]:
        """
        Extract code symbols using regex patterns.

        Args:
            content: Source code content.
            language: Programming language.

        Returns:
            List of extracted code symbols.
        """
        if language == ProgrammingLanguage.PYTHON:
            return self._extract_python_symbols(content)
        elif language in (ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT):
            return self._extract_javascript_symbols(content)
        elif language == ProgrammingLanguage.GO:
            return self._extract_go_symbols(content)
        elif language == ProgrammingLanguage.JAVA:
            return self._extract_java_symbols(content)
        else:
            # Generic extraction for other languages
            return self._extract_generic_symbols(content)

    def _extract_python_symbols(self, content: str) -> list[CodeSymbol]:
        """Extract symbols from Python code."""
        symbols = []
        lines = content.split("\n")

        # Function pattern: def function_name(...):
        func_pattern = re.compile(r"^\s*(async\s+)?def\s+(\w+)\s*\(")
        # Class pattern: class ClassName:
        class_pattern = re.compile(r"^\s*class\s+(\w+)(\(.*?\))?:")

        current_class = None
        current_class_end = 0

        for i, line in enumerate(lines, start=1):
            # Check if we've exited the current class
            if current_class and i > current_class_end:
                current_class = None
                current_class_end = 0

            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                # Find end of class (next class or end of file)
                class_end = self._find_block_end(lines, i - 1)
                symbols.append(
                    CodeSymbol(
                        name=class_name,
                        symbol_type=CodeSymbolType.CLASS,
                        line_start=i,
                        line_end=class_end,
                    )
                )
                current_class = class_name
                current_class_end = class_end
                continue

            # Check for function
            func_match = func_pattern.match(line)
            if func_match:
                func_name = func_match.group(2)
                func_end = self._find_block_end(lines, i - 1)

                # Determine if it's a method or function
                if current_class:
                    symbols.append(
                        CodeSymbol(
                            name=func_name,
                            symbol_type=CodeSymbolType.METHOD,
                            line_start=i,
                            line_end=func_end,
                            parent=current_class,
                        )
                    )
                else:
                    symbols.append(
                        CodeSymbol(
                            name=func_name,
                            symbol_type=CodeSymbolType.FUNCTION,
                            line_start=i,
                            line_end=func_end,
                        )
                    )

        return symbols

    def _extract_javascript_symbols(self, content: str) -> list[CodeSymbol]:
        """Extract symbols from JavaScript/TypeScript code."""
        symbols = []
        lines = content.split("\n")

        # Function patterns
        func_patterns = [
            re.compile(r"^\s*function\s+(\w+)\s*\("),  # function name()
            re.compile(r"^\s*const\s+(\w+)\s*=\s*\(.*?\)\s*=>"),  # const name = () =>
            re.compile(r"^\s*async\s+function\s+(\w+)\s*\("),  # async function
        ]
        # Class pattern
        class_pattern = re.compile(r"^\s*class\s+(\w+)")

        for i, line in enumerate(lines, start=1):
            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                class_end = self._find_brace_block_end(lines, i - 1)
                symbols.append(
                    CodeSymbol(
                        name=class_name,
                        symbol_type=CodeSymbolType.CLASS,
                        line_start=i,
                        line_end=class_end,
                    )
                )
                continue

            # Check for functions
            for pattern in func_patterns:
                func_match = pattern.match(line)
                if func_match:
                    func_name = func_match.group(1)
                    func_end = self._find_brace_block_end(lines, i - 1)
                    symbols.append(
                        CodeSymbol(
                            name=func_name,
                            symbol_type=CodeSymbolType.FUNCTION,
                            line_start=i,
                            line_end=func_end,
                        )
                    )
                    break

        return symbols

    def _extract_go_symbols(self, content: str) -> list[CodeSymbol]:
        """Extract symbols from Go code."""
        symbols = []
        lines = content.split("\n")

        # Function pattern: func name(...) ...
        func_pattern = re.compile(r"^\s*func\s+(?:\(.*?\)\s+)?(\w+)\s*\(")
        # Type pattern: type Name struct
        type_pattern = re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)")

        for i, line in enumerate(lines, start=1):
            # Check for type
            type_match = type_pattern.match(line)
            if type_match:
                type_name = type_match.group(1)
                type_end = self._find_brace_block_end(lines, i - 1)
                symbols.append(
                    CodeSymbol(
                        name=type_name,
                        symbol_type=CodeSymbolType.TYPE,
                        line_start=i,
                        line_end=type_end,
                    )
                )
                continue

            # Check for function
            func_match = func_pattern.match(line)
            if func_match:
                func_name = func_match.group(1)
                func_end = self._find_brace_block_end(lines, i - 1)
                symbols.append(
                    CodeSymbol(
                        name=func_name,
                        symbol_type=CodeSymbolType.FUNCTION,
                        line_start=i,
                        line_end=func_end,
                    )
                )

        return symbols

    def _extract_java_symbols(self, content: str) -> list[CodeSymbol]:
        """Extract symbols from Java code."""
        symbols = []
        lines = content.split("\n")

        # Class pattern
        class_pattern = re.compile(r"^\s*(?:public|private|protected)?\s*class\s+(\w+)")
        # Method pattern
        method_pattern = re.compile(
            r"^\s*(?:public|private|protected|static|final|\s)*\s+\w+\s+(\w+)\s*\("
        )

        for i, line in enumerate(lines, start=1):
            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                class_end = self._find_brace_block_end(lines, i - 1)
                symbols.append(
                    CodeSymbol(
                        name=class_name,
                        symbol_type=CodeSymbolType.CLASS,
                        line_start=i,
                        line_end=class_end,
                    )
                )
                continue

            # Check for method
            method_match = method_pattern.match(line)
            if method_match:
                method_name = method_match.group(1)
                method_end = self._find_brace_block_end(lines, i - 1)
                symbols.append(
                    CodeSymbol(
                        name=method_name,
                        symbol_type=CodeSymbolType.METHOD,
                        line_start=i,
                        line_end=method_end,
                    )
                )

        return symbols

    def _extract_generic_symbols(self, content: str) -> list[CodeSymbol]:
        """Generic symbol extraction for unsupported languages."""
        # Just try to find function-like patterns
        symbols = []
        lines = content.split("\n")
        func_pattern = re.compile(r"^\s*(?:function|def|fn|func)\s+(\w+)")

        for i, line in enumerate(lines, start=1):
            match = func_pattern.match(line)
            if match:
                name = match.group(1)
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=CodeSymbolType.FUNCTION,
                        line_start=i,
                        line_end=i + 10,  # Rough estimate
                    )
                )

        return symbols

    def _find_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a Python block based on indentation."""
        if start_idx >= len(lines):
            return start_idx + 1

        # Get base indentation
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

        # Find where indentation returns to base or less
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent:
                    return i
        return len(lines)

    def _find_brace_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a brace-delimited block."""
        if start_idx >= len(lines):
            return start_idx + 1

        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    found_opening = True
                elif char == "}":
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i + 1

        return len(lines)

    def _symbols_to_sections(self, symbols: list[CodeSymbol], content: str) -> list[Section]:
        """
        Convert code symbols to sections.

        Args:
            symbols: List of code symbols.
            content: Full file content.

        Returns:
            List of sections representing code structure.
        """
        sections = []
        lines = content.split("\n")

        for symbol in symbols:
            # Extract symbol content
            symbol_lines = lines[symbol.line_start - 1 : symbol.line_end]
            symbol_content = "\n".join(symbol_lines)

            # Determine section level based on symbol type
            level = 1 if symbol.symbol_type == CodeSymbolType.CLASS else 2

            sections.append(
                Section(
                    title=f"{symbol.symbol_type.value}: {symbol.name}",
                    level=level,
                    content=symbol_content,
                    start_pos=(symbol.line_start - 1) * 80,  # Rough estimate
                    end_pos=symbol.line_end * 80,
                )
            )

        return sections

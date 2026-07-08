"""Tests for the Python source parser."""

from pathlib import Path

import pytest

from deltx.detection.parser import PythonSourceParser


@pytest.fixture
def parser() -> PythonSourceParser:
    return PythonSourceParser()


class TestValidFunction:
    SOURCE = (
        "def calculate_sum(a: int, b: int) -> int:\n"
        "    # Add two numbers\n"
        "    result = a + b\n"
        "    return result\n"
    )

    def test_identifiers(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("sum.py"))
        for name in ("calculate_sum", "a", "b", "result"):
            assert name in parsed.identifiers

    def test_comment_lines(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("sum.py"))
        assert parsed.comment_lines == 1

    def test_lines_of_code(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("sum.py"))
        assert parsed.lines_of_code == 3

    def test_is_valid(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("sum.py"))
        assert parsed.is_valid is True

    def test_total_lines(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("sum.py"))
        assert parsed.total_lines == 4

    def test_tokens_extracted(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("sum.py"))
        # 'def' keyword survives; comment '#' text does not.
        assert "def" in parsed.tokens
        assert not any(tok.startswith("#") for tok in parsed.tokens)


class TestEmptySource:
    def test_empty_string(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse("", Path("empty.py"))
        assert parsed.is_valid is False
        assert parsed.lines_of_code == 0
        assert parsed.tokens == []
        assert parsed.identifiers == []
        assert parsed.ast_tree is None

    def test_whitespace_only(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse("   \n\n\t\n", Path("blank.py"))
        assert parsed.is_valid is False
        assert parsed.lines_of_code == 0


class TestInvalidSyntax:
    def test_tokens_extracted_despite_syntax_error(
        self, parser: PythonSourceParser
    ) -> None:
        # Lexically tokenizable but not a valid parse (double '=').
        parsed = parser.parse("x = = 5\n", Path("broken.py"))
        assert parsed.is_valid is False
        assert parsed.ast_tree is None
        assert len(parsed.tokens) > 0
        assert "x" in parsed.tokens

    def test_unterminated_construct(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse("def foo(\n", Path("broken.py"))
        assert parsed.is_valid is False


class TestNestedStructures:
    SOURCE = (
        "for i in range(10):\n"
        "    if i > 5:\n"
        "        try:\n"
        "            x = 1\n"
        "        except ValueError:\n"
        "            pass\n"
    )

    def test_depths_exceed_one(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("nested.py"))
        assert parsed.is_valid is True
        assert max(parsed.ast_depths) > 1

    def test_node_types_present(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("nested.py"))
        for node_type in ("For", "If", "Try"):
            assert node_type in parsed.ast_node_types


class TestIdentifierFiltering:
    SOURCE = (
        "import os\n"
        "import math as m\n"
        "\n"
        "def process(data):\n"
        "    result = len(data)\n"
        "    print(result)\n"
        "    return os.getcwd()\n"
    )

    def test_excludes_builtins(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("imp.py"))
        assert "len" not in parsed.identifiers
        assert "print" not in parsed.identifiers

    def test_excludes_import_names(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("imp.py"))
        assert "os" not in parsed.identifiers
        assert "math" not in parsed.identifiers
        assert "m" not in parsed.identifiers

    def test_keeps_user_identifiers(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("imp.py"))
        for name in ("process", "data", "result"):
            assert name in parsed.identifiers


class TestIndentLevels:
    SOURCE = (
        "def f():\n"
        "    if True:\n"
        "        for i in range(3):\n"
        "            x = i\n"
    )

    def test_indent_levels(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse(self.SOURCE, Path("indent.py"))
        assert parsed.indent_levels == [0, 4, 8, 12]

    def test_indent_excludes_comment_lines(
        self, parser: PythonSourceParser
    ) -> None:
        source = "x = 1\n    # indented comment\ny = 2\n"
        parsed = parser.parse(source, Path("c.py"))
        # Comment-only line contributes no indent level.
        assert parsed.indent_levels == [0, 0]


class TestSingleLine:
    def test_single_line_file(self, parser: PythonSourceParser) -> None:
        parsed = parser.parse("x = 42\n", Path("one.py"))
        assert parsed.is_valid is True
        assert parsed.lines_of_code == 1
        assert parsed.total_lines == 1
        assert "x" in parsed.identifiers

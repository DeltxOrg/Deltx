"""Tests for the stylometric feature extractor (F7–F12)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from deltx.detection.features.stylometric import StylometricExtractor
from deltx.detection.models import ParsedSource
from deltx.detection.parser import PythonSourceParser

_FEATURE_KEYS = {
    "f7_avg_identifier_length",
    "f8_identifier_diversity",
    "f9_whitespace_consistency",
    "f10_comment_to_code_ratio",
    "f11_ast_depth_mean",
    "f12_ast_node_type_diversity",
}


@pytest.fixture
def extractor() -> StylometricExtractor:
    return StylometricExtractor()


def _parsed(
    *,
    identifiers: list[str] | None = None,
    indent_levels: list[int] | None = None,
    ast_node_types: list[str] | None = None,
    ast_depths: list[int] | None = None,
    comment_lines: int = 0,
    lines_of_code: int = 1,
    is_valid: bool = True,
) -> ParsedSource:
    """Build a ParsedSource directly, for targeting individual features."""
    return ParsedSource(
        source_code="",
        file_path=Path("stub.py"),
        tokens=[],
        ast_tree=None,
        identifiers=identifiers or [],
        lines_of_code=lines_of_code,
        comment_lines=comment_lines,
        total_lines=lines_of_code + comment_lines,
        indent_levels=indent_levels or [],
        ast_node_types=ast_node_types or [],
        ast_depths=ast_depths or [],
        is_valid=is_valid,
    )


class TestSimpleFunction:
    """End-to-end via the real parser on a small, valid function."""

    SOURCE = "def add(x, y):\n    result = x + y\n    return result\n"

    @pytest.fixture
    def features(self, extractor: StylometricExtractor) -> dict[str, float]:
        parsed = PythonSourceParser().parse(self.SOURCE, Path("add.py"))
        return extractor.extract_features(parsed)

    def test_returns_exactly_six_named_keys(
        self, features: dict[str, float]
    ) -> None:
        assert set(features) == _FEATURE_KEYS
        assert len(features) == 6

    def test_identifier_features_positive(
        self, features: dict[str, float]
    ) -> None:
        assert features["f7_avg_identifier_length"] > 0  # identifiers exist
        assert features["f8_identifier_diversity"] > 0

    def test_ast_depth_positive(self, features: dict[str, float]) -> None:
        assert features["f11_ast_depth_mean"] > 0


class TestIndividualFeatures:
    """Each feature exercised against hand-computed expected values."""

    def test_f7_avg_identifier_length(
        self, extractor: StylometricExtractor
    ) -> None:
        # (5 + 9 + 1) / 3 = 5.0
        parsed = _parsed(identifiers=["hello", "world_var", "x"])
        features = extractor.extract_features(parsed)
        assert features["f7_avg_identifier_length"] == pytest.approx(5.0)

    def test_f7_excludes_underscore_and_dunder(
        self, extractor: StylometricExtractor
    ) -> None:
        # "_" and "__init__" are filtered → mean over ["hello", "x"] = 3.0.
        parsed = _parsed(identifiers=["hello", "x", "_", "__init__"])
        features = extractor.extract_features(parsed)
        assert features["f7_avg_identifier_length"] == pytest.approx(3.0)

    def test_f7_all_filtered_returns_zero(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(identifiers=["_", "__init__", "__repr__"])
        features = extractor.extract_features(parsed)
        assert features["f7_avg_identifier_length"] == 0.0

    def test_f8_identifier_diversity(
        self, extractor: StylometricExtractor
    ) -> None:
        # unique {x, y, z} = 3, total = 5 → 0.6
        parsed = _parsed(identifiers=["x", "x", "y", "y", "z"])
        features = extractor.extract_features(parsed)
        assert features["f8_identifier_diversity"] == pytest.approx(0.6)

    def test_f9_whitespace_consistency_positive(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(indent_levels=[0, 4, 4, 8, 4, 0])
        features = extractor.extract_features(parsed)
        assert features["f9_whitespace_consistency"] > 0

    def test_f9_uniform_indent_is_zero(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(indent_levels=[4, 4, 4, 4])
        features = extractor.extract_features(parsed)
        assert features["f9_whitespace_consistency"] == pytest.approx(0.0)

    def test_f9_single_line_is_zero(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(indent_levels=[0])
        features = extractor.extract_features(parsed)
        assert features["f9_whitespace_consistency"] == 0.0

    def test_f10_comment_to_code_ratio(
        self, extractor: StylometricExtractor
    ) -> None:
        # 2 comment lines / 8 code lines = 0.25
        parsed = _parsed(comment_lines=2, lines_of_code=8)
        features = extractor.extract_features(parsed)
        assert features["f10_comment_to_code_ratio"] == pytest.approx(0.25)

    def test_f11_ast_depth_mean(self, extractor: StylometricExtractor) -> None:
        # mean([0, 1, 1, 2, 3]) = 1.4
        parsed = _parsed(ast_depths=[0, 1, 1, 2, 3])
        features = extractor.extract_features(parsed)
        assert features["f11_ast_depth_mean"] == pytest.approx(1.4)

    def test_f12_node_type_entropy(
        self, extractor: StylometricExtractor
    ) -> None:
        node_types = ["FunctionDef", "Name", "Name", "Return", "Call"]
        # counts: FunctionDef=1, Name=2, Return=1, Call=1; total=5
        expected = -(
            (1 / 5) * math.log2(1 / 5)
            + (2 / 5) * math.log2(2 / 5)
            + (1 / 5) * math.log2(1 / 5)
            + (1 / 5) * math.log2(1 / 5)
        )
        parsed = _parsed(ast_node_types=node_types)
        features = extractor.extract_features(parsed)
        assert features["f12_ast_node_type_diversity"] == pytest.approx(
            expected, abs=0.01
        )

    def test_f12_single_node_type_is_zero(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(ast_node_types=["Name", "Name", "Name"])
        features = extractor.extract_features(parsed)
        assert features["f12_ast_node_type_diversity"] == 0.0


class TestDegenerateInput:
    """Invalid or empty parses must yield an all-zero, well-formed vector."""

    def test_invalid_parse_returns_all_zeros(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(
            identifiers=["x", "y"],
            ast_depths=[0, 1, 2],
            lines_of_code=5,
            is_valid=False,
        )
        features = extractor.extract_features(parsed)
        assert set(features) == _FEATURE_KEYS
        assert all(value == 0.0 for value in features.values())

    def test_zero_loc_returns_all_zeros(
        self, extractor: StylometricExtractor
    ) -> None:
        # A comment-only file: valid parse, but no lines of code.
        source = "# just a comment\n# and another\n"
        parsed = PythonSourceParser().parse(source, Path("comments.py"))
        assert parsed.lines_of_code == 0
        features = extractor.extract_features(parsed)
        assert all(value == 0.0 for value in features.values())

    def test_call_delegates_to_extract_features(
        self, extractor: StylometricExtractor
    ) -> None:
        parsed = _parsed(identifiers=["alpha", "beta"])
        assert extractor(parsed) == extractor.extract_features(parsed)

"""Tests for the distribution feature extractor (F13–F16)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from deltx.detection.features.distribution import DistributionExtractor
from deltx.detection.models import ParsedSource

_FEATURE_KEYS = {
    "f13_shannon_entropy",
    "f14_zipf_coefficient_deviation",
    "f15_bigram_repetition_rate",
    "f16_hapax_legomena_ratio",
}


@pytest.fixture
def extractor() -> DistributionExtractor:
    return DistributionExtractor()


def _parsed(tokens: list[str]) -> ParsedSource:
    """Build a ParsedSource carrying only the token list the features read."""
    return ParsedSource(
        source_code="",
        file_path=Path("stub.py"),
        tokens=tokens,
        ast_tree=None,
        identifiers=[],
        lines_of_code=1,
        comment_lines=0,
        total_lines=1,
        indent_levels=[],
        ast_node_types=[],
        ast_depths=[],
        is_valid=True,
    )


def _tokens_with_frequencies(frequencies: list[int]) -> list[str]:
    """Build a token list where the i-th distinct token occurs frequencies[i] times."""
    tokens: list[str] = []
    for index, frequency in enumerate(frequencies):
        tokens.extend([f"tok{index}"] * frequency)
    return tokens


class TestShannonEntropy:
    """F13 — entropy of the token frequency distribution."""

    def test_matches_hand_computed_value(
        self, extractor: DistributionExtractor
    ) -> None:
        # counts: def=1, foo=2, return=1, x=2; total=6
        tokens = ["def", "foo", "return", "foo", "x", "x"]
        expected = -(2 * (1 / 6) * math.log2(1 / 6) + 2 * (2 / 6) * math.log2(2 / 6))
        features = extractor.extract_features(_parsed(tokens))
        assert features["f13_shannon_entropy"] == pytest.approx(expected, abs=0.01)

    def test_identical_tokens_have_zero_entropy(
        self, extractor: DistributionExtractor
    ) -> None:
        # A single token type carries no uncertainty: p=1, log2(1)=0.
        features = extractor.extract_features(_parsed(["x", "x", "x", "x"]))
        assert features["f13_shannon_entropy"] == pytest.approx(0.0)


class TestZipfDeviation:
    """F14 — deviation of the fitted frequency-rank exponent from α = 1."""

    def test_zipf_distributed_tokens_have_small_deviation(
        self, extractor: DistributionExtractor
    ) -> None:
        # [100, 50, 33, 25, 20] ≈ 100/rank, so the fit should recover α ≈ 1.
        tokens = _tokens_with_frequencies([100, 50, 33, 25, 20])
        features = extractor.extract_features(_parsed(tokens))
        assert features["f14_zipf_coefficient_deviation"] < 0.3

    def test_fewer_than_three_unique_tokens_returns_zero(
        self, extractor: DistributionExtractor
    ) -> None:
        # Two ranks admit an exact fit whose slope says nothing about shape.
        features = extractor.extract_features(_parsed(["a", "a", "b"]))
        assert features["f14_zipf_coefficient_deviation"] == 0.0

    def test_flat_distribution_deviates_by_one(
        self, extractor: DistributionExtractor
    ) -> None:
        # Equal frequencies fit α = 0 — maximally flat, deviation |0 − 1| = 1.
        features = extractor.extract_features(_parsed(["a", "b", "c", "d"]))
        assert features["f14_zipf_coefficient_deviation"] == pytest.approx(1.0)


class TestBigramRepetitionRate:
    """F15 — share of bigram types that occur more than once."""

    def test_matches_hand_computed_value(
        self, extractor: DistributionExtractor
    ) -> None:
        # bigrams: (a,b) x2, (b,a) x1, (b,c) x1 → 1 of 3 types repeats.
        tokens = ["a", "b", "a", "b", "c"]
        features = extractor.extract_features(_parsed(tokens))
        assert features["f15_bigram_repetition_rate"] == pytest.approx(1 / 3)

    def test_single_token_has_no_bigrams(
        self, extractor: DistributionExtractor
    ) -> None:
        features = extractor.extract_features(_parsed(["a"]))
        assert features["f15_bigram_repetition_rate"] == 0.0

    def test_all_distinct_bigrams_rate_is_zero(
        self, extractor: DistributionExtractor
    ) -> None:
        features = extractor.extract_features(_parsed(["a", "b", "c", "d"]))
        assert features["f15_bigram_repetition_rate"] == pytest.approx(0.0)

    def test_fully_repeated_bigrams_rate_is_one(
        self, extractor: DistributionExtractor
    ) -> None:
        # (a,b) and (b,a) each occur twice across "a b a b a".
        features = extractor.extract_features(_parsed(["a", "b", "a", "b", "a"]))
        assert features["f15_bigram_repetition_rate"] == pytest.approx(1.0)


class TestHapaxLegomenaRatio:
    """F16 — share of the vocabulary used exactly once."""

    def test_matches_hand_computed_value(
        self, extractor: DistributionExtractor
    ) -> None:
        # counts: alpha=1, beta=2, gamma=1, delta=2 → hapax {alpha, gamma} = 2
        # of 4 unique tokens.
        tokens = ["alpha", "beta", "gamma", "beta", "delta", "delta"]
        features = extractor.extract_features(_parsed(tokens))
        assert features["f16_hapax_legomena_ratio"] == pytest.approx(0.5)

    def test_all_unique_tokens_ratio_is_one(
        self, extractor: DistributionExtractor
    ) -> None:
        features = extractor.extract_features(_parsed(["a", "b", "c"]))
        assert features["f16_hapax_legomena_ratio"] == pytest.approx(1.0)

    def test_no_hapax_ratio_is_zero(self, extractor: DistributionExtractor) -> None:
        features = extractor.extract_features(_parsed(["a", "a", "b", "b"]))
        assert features["f16_hapax_legomena_ratio"] == pytest.approx(0.0)


class TestDegenerateInput:
    """An empty token list must yield an all-zero, well-formed vector."""

    def test_empty_tokens_return_all_zeros(
        self, extractor: DistributionExtractor
    ) -> None:
        features = extractor.extract_features(_parsed([]))
        assert set(features) == _FEATURE_KEYS
        assert all(value == 0.0 for value in features.values())

    def test_returns_exactly_four_named_keys(
        self, extractor: DistributionExtractor
    ) -> None:
        features = extractor.extract_features(_parsed(["a", "b", "a"]))
        assert set(features) == _FEATURE_KEYS
        assert len(features) == 4

    def test_call_delegates_to_extract_features(
        self, extractor: DistributionExtractor
    ) -> None:
        parsed = _parsed(["a", "b", "a", "c"])
        assert extractor(parsed) == extractor.extract_features(parsed)

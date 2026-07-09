"""Tests for the feature extraction pipeline orchestrator.

Only the first test needs the real language model. Everything else replaces
``pipeline.perplexity`` with a stub returning fixed F1–F6 values, which keeps the
suite offline and lets the stylometric and distribution families be exercised
against real parsed source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deltx.common.config import DeltxConfig
from deltx.detection.models import FeatureVector
from deltx.detection.pipeline import FeatureExtractionPipeline

# Fixed F1–F6 values, distinct and non-zero so a zero-fill is unmistakable.
_STUB_PERPLEXITY: dict[str, float] = {
    "f1_mean_surprisal": 4.5,
    "f2_surprisal_variance": 1.25,
    "f3_sequence_perplexity": 22.6,
    "f4_max_surprisal": 12.0,
    "f5_low_surprisal_ratio": 0.25,
    "f6_surprisal_slope": -0.01,
}

_PERPLEXITY_KEYS = tuple(_STUB_PERPLEXITY)
_STYLOMETRIC_KEYS = tuple(FeatureVector.feature_names()[6:12])
_DISTRIBUTION_KEYS = tuple(FeatureVector.feature_names()[12:16])

VALID_SOURCE = '''def add_numbers(first_value, second_value):
    """Add two numbers together."""
    # Accumulate the running total.
    running_total = first_value + second_value
    return running_total
'''

OTHER_VALID_SOURCE = """class Counter:
    def __init__(self, start):
        self.current_count = start

    def increment(self, step):
        self.current_count += step
        return self.current_count
"""

# Unbalanced paren and a stray colon: defeats both tokenize and ast.parse.
INVALID_SOURCE = """def broken(:
    total = 1 +
    return total
"""

# A source string the stub perplexity extractor is rigged to blow up on.
EXPLODING_SOURCE = "explode_marker = 1 + 2\n"


class _StubPerplexity:
    """Stands in for PerplexityExtractor: no model, fixed output, call log."""

    def __init__(self, *, raise_on: str | None = None) -> None:
        self.raise_on = raise_on
        self.calls: list[str] = []

    def __call__(self, source_code: str) -> dict[str, float]:
        self.calls.append(source_code)
        if self.raise_on is not None and self.raise_on in source_code:
            raise RuntimeError("boom")
        return dict(_STUB_PERPLEXITY)


@pytest.fixture
def pipeline(config: DeltxConfig) -> FeatureExtractionPipeline:
    """A pipeline with the real perplexity extractor (may load the model)."""
    return FeatureExtractionPipeline(config)


@pytest.fixture
def stub_pipeline(config: DeltxConfig) -> FeatureExtractionPipeline:
    """A pipeline whose perplexity family is stubbed; never touches the model."""
    built = FeatureExtractionPipeline(config)
    built.perplexity = _StubPerplexity()  # type: ignore[assignment]
    return built


def _feature_values(vector: FeatureVector, keys: tuple[str, ...]) -> list[float]:
    return [getattr(vector, key) for key in keys]


# -- 1. full integration, real language model --------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("require_model")
def test_full_pipeline_populates_all_sixteen_features(
    pipeline: FeatureExtractionPipeline,
) -> None:
    """A valid file scored by the real LM yields a complete, finite 16-D vector."""
    result = pipeline.extract_file_features(VALID_SOURCE, Path("sample.py"))

    assert result.is_parseable
    assert result.error_message is None
    assert result.lines_of_code == 4

    array = result.feature_vector.to_array()
    assert array.shape == (16,)
    assert np.all(np.isfinite(array)), "no feature may be NaN or infinite"

    # The LM actually ran: surprisal is strictly positive for real source.
    assert result.feature_vector.f1_mean_surprisal > 0.0
    assert result.feature_vector.f3_sequence_perplexity > 1.0


# -- 2. full vector with a mocked perplexity family --------------------------


def test_mocked_perplexity_merges_into_full_vector(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    """F1–F6 come from the stub; F7–F16 are computed for real and are non-zero."""
    result = stub_pipeline.extract_file_features(VALID_SOURCE, Path("sample.py"))

    assert result.is_parseable
    assert result.error_message is None
    assert result.ai_confidence == 0.0  # classifier has not run yet

    for key, expected in _STUB_PERPLEXITY.items():
        assert getattr(result.feature_vector, key) == pytest.approx(expected)

    # The stub saw the raw source, not the parsed form.
    assert stub_pipeline.perplexity.calls == [VALID_SOURCE]  # type: ignore[union-attr]

    assert result.feature_vector.to_array().shape == (16,)
    assert result.feature_vector.f7_avg_identifier_length > 0.0
    assert result.feature_vector.f11_ast_depth_mean > 0.0
    assert result.feature_vector.f13_shannon_entropy > 0.0
    assert result.feature_vector.f16_hapax_legomena_ratio > 0.0


# -- 3. empty source ---------------------------------------------------------


def test_empty_source_is_unparseable_and_all_zero(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    """An empty file zeroes every feature without waking the language model."""
    result = stub_pipeline.extract_file_features("", Path("empty.py"))

    assert result.is_parseable is False
    assert result.lines_of_code == 0
    assert result.error_message == "Source file is empty"
    assert np.all(result.feature_vector.to_array() == 0.0)

    # Short-circuited before the perplexity family, so no model was needed.
    assert stub_pipeline.perplexity.calls == []  # type: ignore[union-attr]


def test_whitespace_only_source_is_treated_as_empty(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    result = stub_pipeline.extract_file_features("   \n\n\t\n", Path("blank.py"))

    assert result.is_parseable is False
    assert np.all(result.feature_vector.to_array() == 0.0)


# -- 4. syntactically invalid source -----------------------------------------


def test_invalid_syntax_keeps_perplexity_features(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    """The LM scores bytes, so F1–F6 survive a parse failure that zeroes F7–F12."""
    result = stub_pipeline.extract_file_features(INVALID_SOURCE, Path("broken.py"))

    assert result.is_parseable is False
    # No *family* raised, so nothing is recorded as an error.
    assert result.error_message is None

    for key, expected in _STUB_PERPLEXITY.items():
        assert getattr(result.feature_vector, key) == pytest.approx(expected)

    # Stylometric features depend on a valid AST and degrade to zero.
    assert _feature_values(result.feature_vector, _STYLOMETRIC_KEYS) == [0.0] * 6


# -- 5. batch processing -----------------------------------------------------


def test_extract_batch_returns_one_result_per_file(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    """Two valid files and one empty file produce three results, in order."""
    files: list[tuple[str, Path]] = [
        (VALID_SOURCE, Path("a.py")),
        (OTHER_VALID_SOURCE, Path("b.py")),
        ("", Path("c.py")),
    ]

    results = stub_pipeline.extract_batch(files)

    assert len(results) == 3
    assert [r.file_path.name for r in results] == ["a.py", "b.py", "c.py"]
    assert results[0].is_parseable
    assert results[1].is_parseable
    assert results[2].is_parseable is False
    assert results[2].error_message == "Source file is empty"


def test_extract_batch_on_empty_list_returns_empty_list(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    assert stub_pipeline.extract_batch([]) == []


# -- 6. failure isolation ----------------------------------------------------


def test_one_failing_file_does_not_sink_the_batch(
    config: DeltxConfig,
) -> None:
    """A raising extractor zeroes only its own family, on only its own file."""
    built = FeatureExtractionPipeline(config)
    built.perplexity = _StubPerplexity(raise_on="explode_marker")  # type: ignore[assignment]

    files: list[tuple[str, Path]] = [
        (VALID_SOURCE, Path("good_first.py")),
        (EXPLODING_SOURCE, Path("bad.py")),
        (OTHER_VALID_SOURCE, Path("good_last.py")),
    ]

    results = built.extract_batch(files)

    assert len(results) == 3
    good_first, bad, good_last = results

    # The failure is recorded, attributed to its family, and contained.
    assert bad.error_message is not None
    assert "perplexity" in bad.error_message
    assert "boom" in bad.error_message
    assert _feature_values(bad.feature_vector, _PERPLEXITY_KEYS) == [0.0] * 6

    # The other two families still ran on the failing file.
    assert bad.is_parseable
    assert bad.feature_vector.f13_shannon_entropy > 0.0
    assert any(_feature_values(bad.feature_vector, _DISTRIBUTION_KEYS))

    # Its neighbours are untouched.
    for neighbour in (good_first, good_last):
        assert neighbour.error_message is None
        assert neighbour.feature_vector.f1_mean_surprisal == pytest.approx(4.5)


def test_family_returning_incomplete_dict_is_treated_as_failure(
    config: DeltxConfig,
) -> None:
    """A short feature dict must not surface as a pydantic error downstream."""

    def _short(source_code: str) -> dict[str, float]:
        return {"f1_mean_surprisal": 1.0}

    built = FeatureExtractionPipeline(config)
    built.perplexity = _short  # type: ignore[assignment]

    result = built.extract_file_features(VALID_SOURCE, Path("short.py"))

    assert result.error_message is not None
    assert "perplexity" in result.error_message
    # Zeroed wholesale, including the one key the family did return.
    assert _feature_values(result.feature_vector, _PERPLEXITY_KEYS) == [0.0] * 6
    assert result.feature_vector.f13_shannon_entropy > 0.0


# -- extract_features_only ---------------------------------------------------


def test_extract_features_only_returns_vector_for_clean_file(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    vector = stub_pipeline.extract_features_only(VALID_SOURCE, Path("sample.py"))

    assert isinstance(vector, FeatureVector)
    assert vector.f1_mean_surprisal == pytest.approx(4.5)
    assert vector.to_array().shape == (16,)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("", "empty source"),
        (INVALID_SOURCE, "unparseable source"),
    ],
)
def test_extract_features_only_returns_none_for_unusable_file(
    stub_pipeline: FeatureExtractionPipeline, source: str, reason: str
) -> None:
    """Dataset construction wants clean samples, not partially-zeroed ones."""
    assert stub_pipeline.extract_features_only(source, Path("bad.py")) is None, reason


def test_extract_features_only_returns_none_when_a_family_fails(
    config: DeltxConfig,
) -> None:
    built = FeatureExtractionPipeline(config)
    built.perplexity = _StubPerplexity(raise_on="explode_marker")  # type: ignore[assignment]

    assert built.extract_features_only(EXPLODING_SOURCE, Path("bad.py")) is None


# -- failures outside the feature families -----------------------------------


class _ExplodingParser:
    """A parser that fails outright, exercising the outer guards."""

    def parse(self, source_code: str, file_path: Path) -> None:
        raise RuntimeError("parser exploded")


@pytest.fixture
def broken_parser_pipeline(config: DeltxConfig) -> FeatureExtractionPipeline:
    built = FeatureExtractionPipeline(config)
    built.perplexity = _StubPerplexity()  # type: ignore[assignment]
    built.parser = _ExplodingParser()  # type: ignore[assignment]
    return built


def test_batch_survives_a_failure_outside_the_feature_families(
    broken_parser_pipeline: FeatureExtractionPipeline,
) -> None:
    """A raise from the parser itself still yields a row, not a crashed batch."""
    results = broken_parser_pipeline.extract_batch([(VALID_SOURCE, Path("a.py"))])

    assert len(results) == 1
    assert results[0].is_parseable is False
    assert results[0].error_message is not None
    assert "parser exploded" in results[0].error_message
    assert np.all(results[0].feature_vector.to_array() == 0.0)


def test_extract_features_only_returns_none_on_unexpected_failure(
    broken_parser_pipeline: FeatureExtractionPipeline,
) -> None:
    assert (
        broken_parser_pipeline.extract_features_only(VALID_SOURCE, Path("a.py")) is None
    )


def test_call_delegates_to_extract_file_features(
    stub_pipeline: FeatureExtractionPipeline,
) -> None:
    called = stub_pipeline(VALID_SOURCE, Path("sample.py"))
    direct = stub_pipeline.extract_file_features(VALID_SOURCE, Path("sample.py"))

    assert called.feature_vector == direct.feature_vector
    assert called.is_parseable == direct.is_parseable

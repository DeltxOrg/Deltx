"""Tests for detection module data models."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from deltx.detection.models import (
    CommitAnalysisResult,
    FeatureVector,
    FileAnalysisResult,
)


def _make_feature_vector(**overrides: float) -> FeatureVector:
    defaults = {name: float(i) for i, name in enumerate(FeatureVector.feature_names())}
    defaults.update(overrides)
    return FeatureVector(**defaults)


def _make_file_result(
    confidence: float, loc: int, path: str = "test.py"
) -> FileAnalysisResult:
    return FileAnalysisResult(
        file_path=Path(path),
        feature_vector=_make_feature_vector(),
        ai_confidence=confidence,
        lines_of_code=loc,
    )


class TestFeatureVector:
    def test_to_array_shape(self) -> None:
        fv = _make_feature_vector()
        arr = fv.to_array()
        assert arr.shape == (16,)
        assert arr.dtype == np.float64

    def test_to_array_order(self) -> None:
        fv = _make_feature_vector()
        arr = fv.to_array()
        for i, name in enumerate(FeatureVector.feature_names()):
            assert arr[i] == getattr(fv, name)

    def test_feature_names_count(self) -> None:
        names = FeatureVector.feature_names()
        assert len(names) == 16

    def test_feature_names_order(self) -> None:
        names = FeatureVector.feature_names()
        for i, name in enumerate(names, start=1):
            assert name.startswith(f"f{i}_")


class TestCommitAnalysisResult:
    def test_aggregate_loc_weighted(self) -> None:
        file_a = _make_file_result(confidence=0.8, loc=100)
        file_b = _make_file_result(confidence=0.2, loc=300)
        result = CommitAnalysisResult.aggregate([file_a, file_b])
        assert result == pytest.approx(35.0)

    def test_aggregate_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CommitAnalysisResult.aggregate([])

    def test_aggregate_zero_loc_raises(self) -> None:
        file_a = _make_file_result(confidence=0.5, loc=0)
        with pytest.raises(ValueError, match="zero"):
            CommitAnalysisResult.aggregate([file_a])


class TestFileAnalysisResult:
    def test_unparseable_with_error(self) -> None:
        result = FileAnalysisResult(
            file_path=Path("broken.py"),
            feature_vector=_make_feature_vector(),
            ai_confidence=0.0,
            lines_of_code=0,
            is_parseable=False,
            error_message="SyntaxError at line 42",
        )
        assert result.is_parseable is False
        assert result.error_message == "SyntaxError at line 42"

    def test_parseable_default(self) -> None:
        result = _make_file_result(confidence=0.5, loc=10)
        assert result.is_parseable is True
        assert result.error_message is None

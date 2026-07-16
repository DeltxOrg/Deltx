"""Tests for dimension scoring and normalization (scoring.py).

Covers: accumulation, density, normalizer fit/transform/save/load,
zero-penalty → 100, max-penalty → 0, unfitted normalizer error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltx.common.exceptions import NormalizerError
from deltx.scoring.models import (
    DimensionScore,
    IsoDimension,
    SonarIssue,
    WeightedIssue,
)
from deltx.scoring.scoring import Normalizer, accumulate, density, module_score


class TestAccumulate:
    """Tests for weight accumulation per dimension."""

    def test_empty_issues(self) -> None:
        """Empty issue list should return zero for all dimensions."""
        result = accumulate([])
        assert all(v == 0.0 for v in result.values())
        assert len(result) == 4  # All four dimensions present.

    def test_single_dimension(self) -> None:
        """Issues in one dimension should only accumulate there."""
        issues = [
            WeightedIssue(
                issue=SonarIssue(rule="r", severity="MAJOR", type="CODE_SMELL", component="a.py"),
                dimension=IsoDimension.MAINTAINABILITY,
                severity_score=4.0,
                weight=10.0,
                frequency=1,
                centrality=0.0,
                churn=0.0,
            ),
            WeightedIssue(
                issue=SonarIssue(rule="r", severity="MINOR", type="CODE_SMELL", component="a.py"),
                dimension=IsoDimension.MAINTAINABILITY,
                severity_score=2.0,
                weight=5.0,
                frequency=1,
                centrality=0.0,
                churn=0.0,
            ),
        ]
        result = accumulate(issues)
        assert result[IsoDimension.MAINTAINABILITY] == 15.0
        assert result[IsoDimension.CORRECTNESS] == 0.0
        assert result[IsoDimension.SECURITY] == 0.0
        assert result[IsoDimension.EFFICIENCY] == 0.0


class TestDensity:
    """Tests for penalty density normalization."""

    def test_zero_loc_returns_zero(self) -> None:
        """Zero LOC should return density 0.0."""
        assert density(10.0, 0) == 0.0

    def test_normal_density(self) -> None:
        """Density should be penalty / LOC."""
        assert density(100.0, 1000) == pytest.approx(0.1)

    def test_negative_loc_returns_zero(self) -> None:
        """Negative LOC should return 0.0."""
        assert density(10.0, -5) == 0.0


class TestNormalizer:
    """Tests for the Z-score normalizer."""

    def test_unfitted_transform_raises(self) -> None:
        """Calling transform on unfitted normalizer should raise."""
        n = Normalizer()
        with pytest.raises(NormalizerError, match="not been fitted"):
            n.transform(0.5, IsoDimension.MAINTAINABILITY)

    def test_unfitted_save_raises(self) -> None:
        """Calling save on unfitted normalizer should raise."""
        n = Normalizer()
        with pytest.raises(NormalizerError, match="Cannot save unfitted"):
            n.save(Path("/tmp/test.json"))

    def test_fit_requires_min_2_observations(self) -> None:
        """Fit should require at least 2 observations per dimension."""
        n = Normalizer()
        with pytest.raises(NormalizerError, match="at least 2"):
            n.fit({
                IsoDimension.MAINTAINABILITY: [0.1],  # Only 1!
                IsoDimension.CORRECTNESS: [0.1, 0.2],
                IsoDimension.SECURITY: [0.1, 0.2],
                IsoDimension.EFFICIENCY: [0.1, 0.2],
            })

    def test_zero_penalty_scores_100(self, fitted_normalizer: Normalizer) -> None:
        """A module with zero penalty density should score 100 (perfect)."""
        score = fitted_normalizer.transform(0.0, IsoDimension.MAINTAINABILITY)
        assert score == pytest.approx(100.0)

    def test_max_penalty_scores_0(self, fitted_normalizer: Normalizer) -> None:
        """A module at the training maximum penalty should score 0."""
        # The fitted normalizer uses max_val=2.0.
        score = fitted_normalizer.transform(2.0, IsoDimension.MAINTAINABILITY)
        assert score == pytest.approx(0.0)

    def test_score_bounded_0_100(self, fitted_normalizer: Normalizer) -> None:
        """Scores should be bounded to [0, 100] even for extreme densities."""
        very_high = fitted_normalizer.transform(999.0, IsoDimension.SECURITY)
        very_low = fitted_normalizer.transform(-999.0, IsoDimension.SECURITY)
        assert 0.0 <= very_high <= 100.0
        assert 0.0 <= very_low <= 100.0

    def test_save_and_load_roundtrip(self, fitted_normalizer: Normalizer, tmp_path: Path) -> None:
        """Save → load should produce identical transform results."""
        path = tmp_path / "normalizer.json"
        fitted_normalizer.save(path)

        loaded = Normalizer()
        loaded.load(path)

        for dim in IsoDimension:
            orig = fitted_normalizer.transform(0.5, dim)
            roundtripped = loaded.transform(0.5, dim)
            assert orig == pytest.approx(roundtripped)

    def test_fingerprint_mismatch_raises(self, fitted_normalizer: Normalizer, tmp_path: Path) -> None:
        """Loading with a mismatched fingerprint should raise."""
        path = tmp_path / "normalizer.json"
        fitted_normalizer.save(path)

        loaded = Normalizer()
        with pytest.raises(NormalizerError, match="fingerprint mismatch"):
            loaded.load(path, expected_fingerprint="wrong_fingerprint")

    def test_load_missing_file_raises(self) -> None:
        """Loading from a non-existent file should raise."""
        n = Normalizer()
        with pytest.raises(NormalizerError, match="not found"):
            n.load(Path("/nonexistent/normalizer.json"))

    def test_is_fitted_property(self, fitted_normalizer: Normalizer) -> None:
        """is_fitted should reflect the normalizer state."""
        fresh = Normalizer()
        assert not fresh.is_fitted
        assert fitted_normalizer.is_fitted


class TestModuleScore:
    """Tests for the module_score function."""

    def test_no_issues_scores_100(self, fitted_normalizer: Normalizer) -> None:
        """A module with no issues should score 100 in all dimensions."""
        scores = module_score([], loc=100, normalizer=fitted_normalizer)
        assert scores.maintainability == pytest.approx(100.0)
        assert scores.correctness == pytest.approx(100.0)
        assert scores.security == pytest.approx(100.0)
        assert scores.efficiency == pytest.approx(100.0)

    def test_issues_lower_score(self, fitted_normalizer: Normalizer) -> None:
        """A module with issues should have lower scores."""
        issues = [
            WeightedIssue(
                issue=SonarIssue(rule="r", severity="BLOCKER", type="BUG", component="a.py"),
                dimension=IsoDimension.CORRECTNESS,
                severity_score=10.0,
                weight=50.0,
                frequency=1,
                centrality=0.5,
                churn=10.0,
            ),
        ]
        scores = module_score(issues, loc=100, normalizer=fitted_normalizer)
        assert scores.correctness < 100.0
        # Other dimensions should still be 100 (no issues there).
        assert scores.maintainability == pytest.approx(100.0)

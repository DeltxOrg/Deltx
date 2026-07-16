"""Tests for Squale exponential aggregation (aggregation.py).

Covers: single module passthrough, all-equal modules, worst-module
dominance, and the canonical [95,95,95,10] test.
"""

from __future__ import annotations

import pytest

from deltx.scoring.aggregation import squale_aggregate, system_scores
from deltx.scoring.models import IsoDimension


class TestSqualeAggregate:
    """Tests for the Squale exponential aggregate function."""

    def test_single_module_passthrough(self) -> None:
        """A single module score should pass through unchanged."""
        assert squale_aggregate([75.0]) == pytest.approx(75.0)

    def test_all_equal_modules(self) -> None:
        """Equal module scores should aggregate to the same score."""
        result = squale_aggregate([80.0, 80.0, 80.0])
        assert result == pytest.approx(80.0)

    def test_worst_module_dominates(self) -> None:
        """The aggregate of [95, 95, 95, 10] must be far below the mean (73.75)."""
        scores = [95.0, 95.0, 95.0, 10.0]
        arithmetic_mean = sum(scores) / len(scores)
        assert arithmetic_mean == pytest.approx(73.75)

        agg = squale_aggregate(scores, lam=30.0)

        # Must be far below the arithmetic mean (at least 20 points).
        assert agg < arithmetic_mean - 20, (
            f"Squale aggregate {agg:.1f} is too close to arithmetic mean {arithmetic_mean}"
        )

        # The worst module must have pulled the aggregate down significantly.
        # The aggregate should be less than 65% of the arithmetic mean.
        assert agg < arithmetic_mean * 0.65, (
            f"Aggregate {agg:.1f} is not dominated enough by worst module "
            f"(should be < {arithmetic_mean * 0.65:.1f})"
        )

    def test_all_perfect_scores_100(self) -> None:
        """All-100 modules should aggregate to 100."""
        assert squale_aggregate([100.0, 100.0, 100.0]) == pytest.approx(100.0)

    def test_all_zero_scores_0(self) -> None:
        """All-0 modules should aggregate to 0."""
        assert squale_aggregate([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_monotonic_in_worst_score(self) -> None:
        """Worsening the worst module should lower the aggregate."""
        agg1 = squale_aggregate([90.0, 90.0, 30.0])
        agg2 = squale_aggregate([90.0, 90.0, 10.0])
        assert agg2 < agg1

    def test_empty_raises(self) -> None:
        """Empty score list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            squale_aggregate([])

    def test_lambda_must_be_gt_1(self) -> None:
        """λ ≤ 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 1"):
            squale_aggregate([50.0], lam=1.0)
        with pytest.raises(ValueError, match="must be > 1"):
            squale_aggregate([50.0], lam=0.5)

    def test_bounded_0_100(self) -> None:
        """Output should always be in [0, 100]."""
        # Extreme case with mixed scores.
        result = squale_aggregate([0.0, 100.0, 50.0, 25.0])
        assert 0.0 <= result <= 100.0

    def test_squale_below_mean_nonconstant(self) -> None:
        """For non-constant vectors, Squale should be strictly below arithmetic mean."""
        scores = [90.0, 70.0, 50.0]
        mean = sum(scores) / len(scores)
        agg = squale_aggregate(scores)
        assert agg < mean, (
            f"Squale {agg:.1f} should be strictly below mean {mean:.1f} "
            f"for non-constant scores"
        )


class TestSystemScores:
    """Tests for system-level aggregation across dimensions."""

    def test_empty_dimension_defaults_to_100(self) -> None:
        """Dimensions with no modules should default to 100."""
        result = system_scores({}, lam=30.0)
        for dim in IsoDimension:
            assert result[dim] == 100.0

    def test_all_dimensions_present(self) -> None:
        """Result should always contain all four dimensions."""
        result = system_scores(
            {IsoDimension.MAINTAINABILITY: [80.0, 70.0]},
            lam=30.0,
        )
        assert len(result) == 4
        for dim in IsoDimension:
            assert dim in result

    def test_per_dimension_aggregation(self) -> None:
        """Each dimension should be aggregated independently."""
        per_module = {
            IsoDimension.MAINTAINABILITY: [90.0, 85.0],
            IsoDimension.CORRECTNESS: [50.0, 20.0],
            IsoDimension.SECURITY: [100.0, 100.0],
            IsoDimension.EFFICIENCY: [70.0, 60.0],
        }
        result = system_scores(per_module, lam=30.0)

        # Security (all 100) should be highest.
        assert result[IsoDimension.SECURITY] == pytest.approx(100.0)
        # Correctness (50, 20) should be lowest.
        assert result[IsoDimension.CORRECTNESS] < result[IsoDimension.MAINTAINABILITY]

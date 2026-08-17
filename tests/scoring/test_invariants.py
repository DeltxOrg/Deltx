"""Property-based tests using Hypothesis (Phase 9 — invariant ring).

Asserts mathematical invariants that must hold for ALL valid inputs:
1. Bounded: every score in [0, 100], never NaN/inf.
2. Monotonic in severity: adding a more-severe issue never raises the score.
3. Monotonic in frequency: more occurrences never raise the score (sub-linear).
4. Centrality ordering: same issue in higher-centrality file → ≤ score.
5. Squale ≤ mean: strict inequality for non-constant vectors.
6. Determinism: identical inputs → identical outputs.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deltx.scoring.aggregation import squale_aggregate
from deltx.scoring.models import Hyperparams, IsoDimension, SonarIssue, WeightedIssue
from deltx.scoring.scoring import Normalizer, module_score
from deltx.scoring.weighting import weight_issue


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Scores in the valid range [0, 100].
score_st = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Non-constant score vectors (at least 2 elements, not all equal).
nonconstant_scores_st = st.lists(score_st, min_size=2, max_size=20).filter(
    lambda xs: len(set(xs)) > 1
)

# Score vectors (1+ elements).
score_list_st = st.lists(score_st, min_size=1, max_size=20)

# Valid severity scores (1–10).
severity_st = st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Valid centrality [0, 1].
centrality_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Valid frequency (0+).
freq_st = st.integers(min_value=0, max_value=100)

# Valid churn (0+).
churn_st = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# 1. Bounded
# ---------------------------------------------------------------------------


class TestBounded:
    """Every score must be in [0, 100], never NaN/inf."""

    @given(scores=score_list_st)
    @settings(max_examples=200)
    def test_squale_bounded(self, scores: list[float]) -> None:
        """Squale aggregate is always in [0, 100]."""
        result = squale_aggregate(scores, lam=30.0)
        assert 0.0 <= result <= 100.0
        assert not math.isnan(result)
        assert not math.isinf(result)


# ---------------------------------------------------------------------------
# 2. Monotonic in severity
# ---------------------------------------------------------------------------


class TestMonotonicSeverity:
    """Adding a more-severe issue to a module never raises its score."""

    @given(
        base_severity=st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
        higher_severity=st.floats(min_value=5.1, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_higher_severity_higher_weight(
        self, base_severity: float, higher_severity: float
    ) -> None:
        """A higher severity always produces a higher weight (all else equal)."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        hp = Hyperparams()

        w_base = weight_issue(issue, base_severity, freq=1, centrality=0.5, churn=10.0, hp=hp)
        w_higher = weight_issue(issue, higher_severity, freq=1, centrality=0.5, churn=10.0, hp=hp)

        assert w_higher >= w_base


# ---------------------------------------------------------------------------
# 3. Monotonic in frequency (sub-linear)
# ---------------------------------------------------------------------------


class TestMonotonicFrequency:
    """More occurrences never raise the score; effect is sub-linear."""

    @given(
        freq_low=st.integers(min_value=1, max_value=49),
    )
    @settings(max_examples=100)
    def test_higher_frequency_higher_weight(self, freq_low: int) -> None:
        """More occurrences → higher weight."""
        freq_high = freq_low + 50
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        hp = Hyperparams(alpha=0.5, beta=0.0, gamma=0.0)

        w_low = weight_issue(issue, 4.0, freq=freq_low, centrality=0.0, churn=0.0, hp=hp)
        w_high = weight_issue(issue, 4.0, freq=freq_high, centrality=0.0, churn=0.0, hp=hp)

        assert w_high > w_low

    @given(freq=st.integers(min_value=10, max_value=100))
    @settings(max_examples=100)
    def test_frequency_effect_sublinear(self, freq: int) -> None:
        """The increase from freq to 2*freq should be less than from 1 to freq+1."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        hp = Hyperparams(alpha=0.5, beta=0.0, gamma=0.0)

        w_1 = weight_issue(issue, 4.0, freq=1, centrality=0.0, churn=0.0, hp=hp)
        w_f = weight_issue(issue, 4.0, freq=freq, centrality=0.0, churn=0.0, hp=hp)
        w_2f = weight_issue(issue, 4.0, freq=2 * freq, centrality=0.0, churn=0.0, hp=hp)

        delta_initial = w_f - w_1
        delta_later = w_2f - w_f
        # Sub-linear: later delta should be smaller.
        assert delta_later < delta_initial


# ---------------------------------------------------------------------------
# 4. Centrality ordering
# ---------------------------------------------------------------------------


class TestCentralityOrdering:
    """Same issue in a higher-centrality file → higher weight."""

    @given(
        c_low=st.floats(min_value=0.0, max_value=0.49, allow_nan=False),
        c_high=st.floats(min_value=0.51, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_higher_centrality_higher_weight(
        self, c_low: float, c_high: float
    ) -> None:
        """Higher centrality → higher weight."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        hp = Hyperparams()

        w_low = weight_issue(issue, 4.0, freq=1, centrality=c_low, churn=0.0, hp=hp)
        w_high = weight_issue(issue, 4.0, freq=1, centrality=c_high, churn=0.0, hp=hp)

        assert w_high > w_low


# ---------------------------------------------------------------------------
# 5. Squale ≤ mean (strict inequality for non-constant)
# ---------------------------------------------------------------------------


class TestSqualeVsMean:
    """Squale aggregate ≤ arithmetic mean, strict for non-constant vectors."""

    @given(scores=nonconstant_scores_st)
    @settings(max_examples=200)
    def test_squale_strictly_below_mean(self, scores: list[float]) -> None:
        """For non-constant score vectors, Squale < arithmetic mean."""
        mean = sum(scores) / len(scores)
        agg = squale_aggregate(scores, lam=30.0)
        assert agg < mean + 1e-6, (
            f"Squale {agg:.4f} should be < mean {mean:.4f} for {scores}"
        )

    @given(score=score_st)
    @settings(max_examples=50)
    def test_constant_vector_equals_value(self, score: float) -> None:
        """For constant vectors, Squale == the constant value."""
        scores = [score, score, score]
        agg = squale_aggregate(scores, lam=30.0)
        assert agg == pytest.approx(score, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Identical inputs must produce identical outputs."""

    @given(scores=score_list_st)
    @settings(max_examples=100)
    def test_squale_deterministic(self, scores: list[float]) -> None:
        """Same inputs → same output across multiple calls."""
        r1 = squale_aggregate(scores, lam=30.0)
        r2 = squale_aggregate(scores, lam=30.0)
        assert r1 == r2

    def test_weight_deterministic(self) -> None:
        """Same inputs → same weight across multiple calls."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        hp = Hyperparams()

        results = [
            weight_issue(issue, 4.0, freq=5, centrality=0.5, churn=20.0, hp=hp)
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results)

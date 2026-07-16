"""Tests for dynamic issue weighting (weighting.py).

Covers: zero-input collapse to base severity, sub-linear frequency effect,
centrality ordering, and negative/NaN input rejection.
"""

from __future__ import annotations

import math

import pytest

from deltx.scoring.models import Hyperparams, IsoDimension, SonarIssue
from deltx.scoring.weighting import weight_all, weight_issue


class TestWeightIssue:
    """Tests for the per-issue weighting formula."""

    def test_zero_frequency_centrality_churn(self) -> None:
        """With zero freq, centrality, and churn, weight = base severity."""
        issue = SonarIssue(
            rule="python:S1234", severity="MAJOR", type="CODE_SMELL", component="a.py"
        )
        hp = Hyperparams()
        # freq=0, centrality=0, churn=0
        w = weight_issue(issue, severity_score=4.0, freq=0, centrality=0.0, churn=0.0, hp=hp)

        # w = 4 * (1 + 0.5*ln(1+0)) * (1 + 1.0*0) * (1 + 0.3*0) = 4 * 1 * 1 * 1 = 4
        assert w == pytest.approx(4.0)

    def test_frequency_raises_weight_sublinearly(self) -> None:
        """Increasing frequency should raise weight sub-linearly (log check).

        Sub-linearity means the marginal weight increase per additional
        occurrence decreases as frequency grows.
        """
        issue = SonarIssue(
            rule="python:S1234", severity="MAJOR", type="CODE_SMELL", component="a.py"
        )
        hp = Hyperparams(alpha=0.5, beta=0.0, gamma=0.0)

        w_base = weight_issue(issue, 4.0, freq=0, centrality=0.0, churn=0.0, hp=hp)
        w_low = weight_issue(issue, 4.0, freq=10, centrality=0.0, churn=0.0, hp=hp)
        w_high = weight_issue(issue, 4.0, freq=100, centrality=0.0, churn=0.0, hp=hp)

        # Weight should increase with frequency.
        assert w_low > w_base
        assert w_high > w_low

        # Sub-linearity: marginal increase per unit frequency should decrease.
        # weight(10) / 10 > weight(100) / 100 (in relative terms)
        marginal_low = (w_low - w_base) / 10     # avg increase per freq unit, 0→10
        marginal_high = (w_high - w_low) / 90     # avg increase per freq unit, 10→100
        assert marginal_high < marginal_low, (
            f"Frequency effect should be sub-linear: "
            f"marginal(0→10)={marginal_low:.6f}, marginal(10→100)={marginal_high:.6f}"
        )

    def test_high_centrality_outweighs_low(self) -> None:
        """Same issue in a high-centrality file should weigh more."""
        issue = SonarIssue(
            rule="python:S1234", severity="MAJOR", type="CODE_SMELL", component="a.py"
        )
        hp = Hyperparams()

        w_low = weight_issue(issue, 4.0, freq=1, centrality=0.1, churn=0.0, hp=hp)
        w_high = weight_issue(issue, 4.0, freq=1, centrality=0.9, churn=0.0, hp=hp)

        assert w_high > w_low

    def test_churn_increases_weight(self) -> None:
        """Higher churn should increase weight."""
        issue = SonarIssue(
            rule="python:S1234", severity="MAJOR", type="CODE_SMELL", component="a.py"
        )
        hp = Hyperparams()

        w_low = weight_issue(issue, 4.0, freq=1, centrality=0.0, churn=10.0, hp=hp)
        w_high = weight_issue(issue, 4.0, freq=1, centrality=0.0, churn=100.0, hp=hp)

        assert w_high > w_low

    def test_negative_severity_raises(self) -> None:
        """Negative severity score should raise ValueError."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        with pytest.raises(ValueError, match="Negative"):
            weight_issue(issue, -1.0, freq=0, centrality=0.0, churn=0.0, hp=Hyperparams())

    def test_negative_centrality_raises(self) -> None:
        """Negative centrality should raise ValueError."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        with pytest.raises(ValueError, match="Negative"):
            weight_issue(issue, 4.0, freq=0, centrality=-0.1, churn=0.0, hp=Hyperparams())

    def test_nan_severity_raises(self) -> None:
        """NaN severity should raise ValueError."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        with pytest.raises(ValueError, match="NaN"):
            weight_issue(issue, float("nan"), freq=0, centrality=0.0, churn=0.0, hp=Hyperparams())

    def test_nan_centrality_raises(self) -> None:
        """NaN centrality should raise ValueError."""
        issue = SonarIssue(rule="r", severity="X", type="BUG", component="a.py")
        with pytest.raises(ValueError, match="NaN"):
            weight_issue(issue, 4.0, freq=0, centrality=float("nan"), churn=0.0, hp=Hyperparams())


class TestWeightAll:
    """Tests for bulk issue weighting."""

    def test_frequency_computed_per_rule_per_file(self) -> None:
        """Frequency should be computed per-rule-per-file."""
        issues = [
            SonarIssue(rule="python:S1234", severity="MAJOR", type="CODE_SMELL",
                       component="a.py", line=1),
            SonarIssue(rule="python:S1234", severity="MAJOR", type="CODE_SMELL",
                       component="a.py", line=2),
            SonarIssue(rule="python:S1234", severity="MAJOR", type="CODE_SMELL",
                       component="b.py", line=1),
        ]

        weighted = weight_all(
            issues,
            centrality_fn=lambda _: 0.0,
            churn_fn=lambda _: 0.0,
        )

        # a.py has freq=2 for this rule, b.py has freq=1.
        a_issues = [w for w in weighted if w.issue.component == "a.py"]
        b_issues = [w for w in weighted if w.issue.component == "b.py"]

        assert all(w.frequency == 2 for w in a_issues)
        assert all(w.frequency == 1 for w in b_issues)
        # a.py issues should have higher weight due to higher frequency.
        assert a_issues[0].weight > b_issues[0].weight

    def test_empty_issues(self) -> None:
        """Empty issue list should produce empty weighted list."""
        weighted = weight_all([], lambda _: 0.0, lambda _: 0.0)
        assert weighted == []

    def test_all_issues_classified(self, sample_issues: list[SonarIssue]) -> None:
        """All fixture issues should be classified and weighted."""
        weighted = weight_all(
            sample_issues,
            centrality_fn=lambda _: 0.5,
            churn_fn=lambda _: 10.0,
        )
        assert len(weighted) == len(sample_issues)
        assert all(w.weight > 0 for w in weighted)

"""Tests for ISO/IEC 25010 dimension mapping (iso_mapping.py).

Covers: type-based mapping, rule-key overrides, severity scoring,
fallback bucket with warning log, and exhaustive fixture coverage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from deltx.scoring.iso_mapping import (
    SEVERITY_SCORES,
    classify,
    severity_to_score,
)
from deltx.scoring.models import IsoDimension, SonarIssue


class TestSeverityMapping:
    """Tests for severity → numeric score mapping."""

    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            ("BLOCKER", 10.0),
            ("CRITICAL", 7.0),
            ("MAJOR", 4.0),
            ("MINOR", 2.0),
            ("INFO", 1.0),
        ],
    )
    def test_known_severities(self, severity: str, expected: float) -> None:
        """Each known severity maps to its documented numeric score."""
        assert severity_to_score(severity) == expected

    def test_case_insensitive(self) -> None:
        """Severity mapping should be case-insensitive."""
        assert severity_to_score("blocker") == 10.0
        assert severity_to_score("Critical") == 7.0

    def test_unknown_severity_defaults_to_info(self) -> None:
        """Unknown severity should default to INFO (1.0) with a warning."""
        assert severity_to_score("UNKNOWN") == 1.0


class TestTypeBasedMapping:
    """Tests for the default type → dimension mapping."""

    @pytest.mark.parametrize(
        ("issue_type", "expected_dim"),
        [
            ("BUG", IsoDimension.CORRECTNESS),
            ("VULNERABILITY", IsoDimension.SECURITY),
            ("SECURITY_HOTSPOT", IsoDimension.SECURITY),
            ("CODE_SMELL", IsoDimension.MAINTAINABILITY),
        ],
    )
    def test_type_to_dimension(self, issue_type: str, expected_dim: IsoDimension) -> None:
        """Each standard issue type maps to its expected dimension."""
        issue = SonarIssue(rule="python:S9999", severity="MAJOR", type=issue_type, component="a.py")
        dim, _ = classify(issue)
        assert dim == expected_dim


class TestRuleOverrides:
    """Tests for rule-key override mappings."""

    @pytest.mark.parametrize(
        "rule",
        [
            "python:S5765",  # Unnecessary comprehension → efficiency
            "python:S1481",  # Unused local variable → efficiency
            "python:S3776",  # Cognitive complexity → efficiency
        ],
    )
    def test_efficiency_overrides(self, rule: str) -> None:
        """Efficiency-semantic CODE_SMELLs should override to EFFICIENCY."""
        issue = SonarIssue(rule=rule, severity="MINOR", type="CODE_SMELL", component="a.py")
        dim, _ = classify(issue)
        assert dim == IsoDimension.EFFICIENCY

    @pytest.mark.parametrize(
        "rule",
        [
            "python:S5607",  # Unreachable code → correctness
            "python:S1763",  # All paths return → correctness
        ],
    )
    def test_correctness_overrides(self, rule: str) -> None:
        """Correctness-semantic CODE_SMELLs should override to CORRECTNESS."""
        issue = SonarIssue(rule=rule, severity="MAJOR", type="CODE_SMELL", component="a.py")
        dim, _ = classify(issue)
        assert dim == IsoDimension.CORRECTNESS


class TestFallbackBucket:
    """Tests for the fallback mapping for unknown types."""

    def test_unknown_type_falls_back_to_maintainability(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown issue type should fall back to MAINTAINABILITY with a warning."""
        issue = SonarIssue(rule="python:UNKNOWN", severity="MAJOR", type="WEIRD_TYPE", component="a.py")

        with caplog.at_level(logging.WARNING):
            dim, score = classify(issue)

        assert dim == IsoDimension.MAINTAINABILITY
        assert score == 4.0  # MAJOR
        assert "Unmapped issue type" in caplog.text


class TestFixtureClassification:
    """Test that every rule in the fixture resolves to exactly one dimension."""

    def test_all_fixture_rules_resolve(self, sample_issues: list[SonarIssue]) -> None:
        """Every issue in the fixture must classify to exactly one dimension."""
        for issue in sample_issues:
            dim, score = classify(issue)
            assert isinstance(dim, IsoDimension), f"Rule {issue.rule} did not resolve"
            assert score > 0, f"Rule {issue.rule} got zero severity score"

    def test_fixture_dimensions_are_diverse(self, sample_issues: list[SonarIssue]) -> None:
        """The fixture should exercise at least 3 of the 4 dimensions."""
        dims = {classify(issue)[0] for issue in sample_issues}
        assert len(dims) >= 3, f"Only got dimensions: {dims}"

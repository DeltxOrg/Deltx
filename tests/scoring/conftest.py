"""Shared pytest fixtures for the scoring module tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltx.scoring.models import (
    Hyperparams,
    IsoDimension,
    SonarIssue,
    SonarMeasures,
)
from deltx.scoring.scoring import Normalizer

# Path to the fixtures directory shipped with the scoring module.
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "deltx" / "scoring" / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the scoring fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_issues_path(fixtures_dir: Path) -> Path:
    """Return the path to the sample issues fixture."""
    return fixtures_dir / "sample_issues.json"


@pytest.fixture
def sample_measures_path(fixtures_dir: Path) -> Path:
    """Return the path to the sample measures fixture."""
    return fixtures_dir / "sample_measures.json"


@pytest.fixture
def sample_issues(sample_issues_path: Path) -> list[SonarIssue]:
    """Load and parse sample SonarQube issues from the fixture."""
    with open(sample_issues_path) as f:
        data = json.load(f)
    return [
        SonarIssue(
            rule=raw["rule"],
            severity=raw.get("severity", "INFO"),
            type=raw.get("type", "CODE_SMELL"),
            component=raw.get("component", ""),
            line=raw.get("line", 0),
            effort=raw.get("effort", "0min"),
            message=raw.get("message", ""),
        )
        for raw in data["issues"]
    ]


@pytest.fixture
def sample_measures(sample_measures_path: Path) -> SonarMeasures:
    """Load sample SonarQube measures from the fixture."""
    with open(sample_measures_path) as f:
        data = json.load(f)
    measures = data["component"]["measures"]
    result: dict[str, str] = {}
    for m in measures:
        result[m["metric"]] = m["value"]
    return SonarMeasures(
        ncloc=int(result.get("ncloc", 0)),
        complexity=int(result.get("complexity", 0)),
        cognitive_complexity=int(result.get("cognitive_complexity", 0)),
        duplicated_lines_density=float(result.get("duplicated_lines_density", 0.0)),
    )


@pytest.fixture
def default_hyperparams() -> Hyperparams:
    """Default hyperparameters."""
    return Hyperparams()


@pytest.fixture
def fitted_normalizer() -> Normalizer:
    """A normalizer fitted on synthetic training data."""
    normalizer = Normalizer()
    normalizer.fit({
        IsoDimension.MAINTAINABILITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        IsoDimension.CORRECTNESS: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        IsoDimension.SECURITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        IsoDimension.EFFICIENCY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    })
    return normalizer


def make_issue(
    rule: str = "python:S1234",
    severity: str = "MAJOR",
    issue_type: str = "CODE_SMELL",
    component: str = "src/module.py",
    line: int = 10,
) -> SonarIssue:
    """Factory for creating SonarIssue instances in tests."""
    return SonarIssue(
        rule=rule,
        severity=severity,
        type=issue_type,
        component=component,
        line=line,
    )

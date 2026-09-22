"""Dimension mapping, fixture scoring and maintainability synthetic factors."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from deltx.scoring.config import RuleMapping, ScoringConfig
from deltx.scoring.models import (
    Dimension,
    IssueImpact,
    IssueType,
    Severity,
    SonarIssue,
    SonarMeasures,
)
from deltx.scoring.squale.engine import map_issue, score_checkpoint


def issue(
    kind: IssueType = IssueType.BUG, severity: Severity = Severity.BLOCKER
) -> SonarIssue:
    return SonarIssue("i", "python:test", severity, kind, Path("a.py"))


def measures(**values: float) -> SonarMeasures:
    return SonarMeasures.model_validate(
        {
            "ncloc": 1000,
            "cognitive_complexity": 0,
            "duplicated_lines_density": 0,
            "sqale_index": 0,
            **values,
        }
    )


@pytest.mark.parametrize(
    "kind,dimension",
    [
        (IssueType.BUG, Dimension.CORRECTNESS),
        (IssueType.VULNERABILITY, Dimension.SECURITY),
        (IssueType.SECURITY_HOTSPOT, Dimension.SECURITY),
        (IssueType.CODE_SMELL, Dimension.MAINTAINABILITY),
    ],
)
def test_fallback(kind: IssueType, dimension: Dimension) -> None:
    assert map_issue(issue(kind), ScoringConfig()) == {dimension: 1}


def test_override_multidimension_efficiency() -> None:
    coefficients = {Dimension.EFFICIENCY: 1.0, Dimension.CORRECTNESS: 0.5}
    config = ScoringConfig(
        rule_overrides=(RuleMapping(rule="python:test", coefficients=coefficients),)
    )
    assert map_issue(issue(IssueType.CODE_SMELL), config) == coefficients
    scores = score_checkpoint((issue(),), measures(), {}, {}, config)
    assert scores.efficiency < scores.correctness < 100
    assert scores.maintainability == 100


def test_unknown_observable_and_no_issues() -> None:
    result = score_checkpoint(
        (issue(IssueType.UNKNOWN),), measures(), {}, {}, ScoringConfig()
    )
    assert result.unmapped_rules == ("python:test",)
    assert result.correctness == result.security == result.efficiency == 100
    assert result.maintainability == 100


def test_blocker_not_masked_by_minor_fixture() -> None:
    config = ScoringConfig()
    blocker = issue()
    minor = issue(severity=Severity.MINOR)
    solo = score_checkpoint((blocker,), measures(), {}, {}, config)
    only_minor = score_checkpoint((minor,) * 30, measures(), {}, {}, config)
    combined = score_checkpoint((blocker, *([minor] * 30)), measures(), {}, {}, config)
    assert solo.correctness < 50
    assert combined.correctness < only_minor.correctness - 1
    assert 0 <= combined.correctness <= 100


@pytest.mark.parametrize(
    "metric", ["sqale_index", "cognitive_complexity", "duplicated_lines_density"]
)
def test_maintainability_responds_to_each_metric(metric: str) -> None:
    clean = score_checkpoint((), measures(), {}, {}, ScoringConfig())
    poor = score_checkpoint((), measures(**{metric: 100}), {}, {}, ScoringConfig())
    assert poor.maintainability < clean.maintainability


def test_zero_ncloc_and_mapping_validation() -> None:
    result = score_checkpoint((), measures(ncloc=0), {}, {}, ScoringConfig())
    assert result.maintainability == 100
    with pytest.raises(ValidationError):
        RuleMapping(rule="python:test", coefficients={Dimension.SECURITY: -1})
    with pytest.raises(ValidationError):
        RuleMapping(rule="python:test", coefficients={})


def test_mqr_scores_each_quality_with_its_own_severity() -> None:
    finding = SonarIssue(
        "a",
        "python:multi",
        Severity.BLOCKER,
        IssueType.CODE_SMELL,
        Path("a.py"),
        (
            IssueImpact(Dimension.SECURITY, Severity.BLOCKER),
            IssueImpact(Dimension.CORRECTNESS, Severity.MINOR),
        ),
    )
    config = ScoringConfig()
    assert map_issue(finding, config) == {
        Dimension.SECURITY: 1,
        Dimension.CORRECTNESS: 1,
    }
    result = score_checkpoint((finding,), measures(), {}, {}, config)
    assert result.security < result.correctness < 100
    assert result.maintainability == result.efficiency == 100


def test_override_retains_efficiency_when_mqr_impacts_are_available() -> None:
    finding = SonarIssue(
        "a",
        "python:S2190",
        Severity.BLOCKER,
        IssueType.BUG,
        Path("a.py"),
        (IssueImpact(Dimension.CORRECTNESS, Severity.BLOCKER),),
    )
    result = score_checkpoint((finding,), measures(), {}, {}, ScoringConfig())
    assert result.correctness == result.efficiency < 100

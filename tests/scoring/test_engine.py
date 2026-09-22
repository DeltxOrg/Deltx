"""Dimension mapping, fixture scoring and maintainability synthetic factors."""

import math
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest
from pydantic import ValidationError

from deltx.common.exceptions import ConfigurationError
from deltx.scoring.config import (
    DimensionConfig,
    MetricConfig,
    RuleMapping,
    ScoringConfig,
)
from deltx.scoring.models import (
    CleanCodeAttribute,
    Dimension,
    IssueImpact,
    IssueType,
    RuleCatalog,
    Severity,
    SonarIssue,
    SonarMeasures,
    SonarRuleMetadata,
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


def catalog(
    attribute: CleanCodeAttribute = CleanCodeAttribute.LOGICAL,
    rule_key: str = "python:test",
) -> RuleCatalog:
    return RuleCatalog(
        {
            rule_key: SonarRuleMetadata(rule_key, attribute),
            "python:efficient": SonarRuleMetadata(
                "python:efficient", CleanCodeAttribute.EFFICIENT
            ),
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
    assert map_issue(issue(kind), catalog().get("python:test"), ScoringConfig()) == (
        IssueImpact(dimension, Severity.BLOCKER),
    )


def test_coefficients_adjust_weight_without_replacing_mappings() -> None:
    coefficients = {Dimension.EFFICIENCY: 1.0, Dimension.CORRECTNESS: 0.5}
    config = ScoringConfig(
        rule_overrides=(RuleMapping(rule="python:test", coefficients=coefficients),)
    )
    rules = catalog(CleanCodeAttribute.EFFICIENT)
    scores = score_checkpoint((issue(),), measures(), {}, {}, config, rules)
    assert scores.efficiency < scores.correctness < 100
    assert scores.maintainability == 100
    # Configured coefficients cannot manufacture efficiency for a LOGICAL rule.
    with pytest.raises(ConfigurationError, match="cannot add dimensions"):
        score_checkpoint((issue(),), measures(), {}, {}, config, catalog())


def test_unknown_observable_and_no_issues() -> None:
    result = score_checkpoint(
        (issue(IssueType.UNKNOWN),), measures(), {}, {}, ScoringConfig(), catalog()
    )
    assert result.unmapped_rules == ("python:test",)
    assert result.correctness == result.security == result.efficiency == 100
    assert result.maintainability == 100


def test_blocker_not_masked_by_minor_fixture() -> None:
    config = ScoringConfig()
    blocker = issue()
    minor = issue(severity=Severity.MINOR)
    solo = score_checkpoint((blocker,), measures(), {}, {}, config, catalog())
    only_minor = score_checkpoint((minor,) * 30, measures(), {}, {}, config, catalog())
    combined = score_checkpoint(
        (blocker, *([minor] * 30)), measures(), {}, {}, config, catalog()
    )
    assert solo.correctness < 50
    assert combined.correctness < only_minor.correctness - 1
    assert 0 <= combined.correctness <= 100


@pytest.mark.parametrize(
    "metric", ["sqale_index", "cognitive_complexity", "duplicated_lines_density"]
)
def test_maintainability_responds_to_each_metric(metric: str) -> None:
    clean = score_checkpoint((), measures(), {}, {}, ScoringConfig(), catalog())
    poor = score_checkpoint(
        (), measures(**{metric: 100}), {}, {}, ScoringConfig(), catalog()
    )
    assert poor.maintainability < clean.maintainability
    assert poor.correctness == poor.security == poor.efficiency == 100


def test_zero_ncloc_and_mapping_validation() -> None:
    result = score_checkpoint((), measures(ncloc=0), {}, {}, ScoringConfig(), catalog())
    assert result.maintainability == 100
    with pytest.raises(ValidationError):
        RuleMapping(rule="python:test", coefficients={Dimension.SECURITY: -1})
    with pytest.raises(ValidationError):
        RuleMapping(rule="python:test", coefficients={})
    with pytest.raises(ValidationError, match="EFFICIENT rule metadata"):
        ScoringConfig(fallback_mapping={IssueType.BUG: Dimension.EFFICIENCY})


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
    rules = catalog(rule_key=finding.rule)
    assert set(map_issue(finding, rules.get(finding.rule), config)) == {
        IssueImpact(Dimension.SECURITY, Severity.BLOCKER),
        IssueImpact(Dimension.CORRECTNESS, Severity.MINOR),
    }
    result = score_checkpoint((finding,), measures(), {}, {}, config, rules)
    assert result.security < result.correctness < 100
    assert result.maintainability == result.efficiency == 100


def test_infinite_recursion_does_not_imply_efficiency() -> None:
    finding = SonarIssue(
        "a",
        "python:S2190",
        Severity.BLOCKER,
        IssueType.BUG,
        Path("a.py"),
        (IssueImpact(Dimension.CORRECTNESS, Severity.BLOCKER),),
    )
    result = score_checkpoint(
        (finding,), measures(), {}, {}, ScoringConfig(), catalog(rule_key=finding.rule)
    )
    assert result.correctness < 100
    assert result.efficiency == 100


@pytest.mark.parametrize(
    "impacts,attribute,expected",
    [
        (
            (IssueImpact(Dimension.MAINTAINABILITY, Severity.CRITICAL),),
            CleanCodeAttribute.CLEAR,
            {Dimension.MAINTAINABILITY: 4},
        ),
        (
            (IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),),
            CleanCodeAttribute.LOGICAL,
            {Dimension.CORRECTNESS: 4},
        ),
        (
            (IssueImpact(Dimension.SECURITY, Severity.BLOCKER),),
            CleanCodeAttribute.TRUSTWORTHY,
            {Dimension.SECURITY: 5},
        ),
        (
            (IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),),
            CleanCodeAttribute.EFFICIENT,
            {Dimension.CORRECTNESS: 4, Dimension.EFFICIENCY: 4},
        ),
        (
            (
                IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),
                IssueImpact(Dimension.MAINTAINABILITY, Severity.MAJOR),
            ),
            CleanCodeAttribute.EFFICIENT,
            {
                Dimension.CORRECTNESS: 4,
                Dimension.MAINTAINABILITY: 3,
                Dimension.EFFICIENCY: 4,
            },
        ),
        (
            (
                IssueImpact(Dimension.SECURITY, Severity.BLOCKER),
                IssueImpact(Dimension.MAINTAINABILITY, Severity.MINOR),
            ),
            CleanCodeAttribute.EFFICIENT,
            {
                Dimension.SECURITY: 5,
                Dimension.MAINTAINABILITY: 2,
                Dimension.EFFICIENCY: 5,
            },
        ),
    ],
)
def test_required_modern_mappings(
    impacts: tuple[IssueImpact, ...],
    attribute: CleanCodeAttribute,
    expected: dict[Dimension, int],
) -> None:
    # Deliberately conflicting legacy fields and rule defaults must not win.
    finding = replace(issue(IssueType.VULNERABILITY, Severity.INFO), impacts=impacts)
    rule = SonarRuleMetadata(
        finding.rule, attribute, (IssueImpact(Dimension.SECURITY, Severity.BLOCKER),)
    )
    actual = map_issue(finding, rule, ScoringConfig())
    assert {i.dimension: i.severity.weight for i in actual} == expected
    assert len(actual) == len(expected)


def test_duplicate_impacts_are_order_independent_and_scored_once() -> None:
    impacts = (
        IssueImpact(Dimension.CORRECTNESS, Severity.MAJOR),
        IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),
        IssueImpact(Dimension.MAINTAINABILITY, Severity.MINOR),
    )
    rules = catalog(CleanCodeAttribute.EFFICIENT)
    canonical = replace(issue(), impacts=impacts[1:])
    expected = score_checkpoint(
        (canonical,), measures(), {}, {}, ScoringConfig(), rules
    )
    for ordering in permutations(impacts):
        finding = replace(issue(), impacts=ordering)
        mapped = map_issue(finding, rules.get(finding.rule), ScoringConfig())
        assert {i.dimension: i.severity.weight for i in mapped} == {
            Dimension.CORRECTNESS: 4,
            Dimension.MAINTAINABILITY: 2,
            Dimension.EFFICIENCY: 4,
        }
        assert (
            score_checkpoint((finding,), measures(), {}, {}, ScoringConfig(), rules)
            == expected
        )


@pytest.mark.parametrize("severity", list(Severity))
def test_efficient_legacy_issue_uses_its_valid_legacy_severity(
    severity: Severity,
) -> None:
    finding = issue(IssueType.UNKNOWN, severity)
    assert map_issue(
        finding,
        catalog(CleanCodeAttribute.EFFICIENT).get(finding.rule),
        ScoringConfig(),
    ) == (IssueImpact(Dimension.EFFICIENCY, severity),)


def test_efficiency_coverage_and_missing_metadata_fail_explicitly() -> None:
    for rules in (
        RuleCatalog(),
        RuleCatalog(
            {
                "python:test": SonarRuleMetadata(
                    "python:test", CleanCodeAttribute.LOGICAL
                )
            }
        ),
    ):
        with pytest.raises(ConfigurationError, match="no EFFICIENT rules"):
            score_checkpoint((), measures(), {}, {}, ScoringConfig(), rules)
    with pytest.raises(ConfigurationError, match="missing active Python rule metadata"):
        score_checkpoint(
            (replace(issue(), rule="python:missing"),),
            measures(),
            {},
            {},
            ScoringConfig(),
            catalog(),
        )
    with pytest.raises(ConfigurationError, match="does not match"):
        map_issue(issue(), catalog().get("python:efficient"), ScoringConfig())
    with pytest.raises(ConfigurationError, match="no EFFICIENCY software quality"):
        map_issue(
            replace(
                issue(), impacts=(IssueImpact(Dimension.EFFICIENCY, Severity.MAJOR),)
            ),
            catalog().get("python:test"),
            ScoringConfig(),
        )


def test_four_practice_composite_uses_nonlinear_weighted_squale() -> None:
    # rho=0 isolates the known severity mark from contextual normalization.
    config = ScoringConfig(
        dimensions={d: DimensionConfig(rho=0) for d in Dimension},
        maintainability_issue_omega=2,
        debt=MetricConfig(tau=60, omega=3),
        complexity=MetricConfig(tau=20, omega=4),
        duplication=MetricConfig(tau=5, omega=5),
    )
    finding = issue(IssueType.CODE_SMELL, Severity.MAJOR)
    observed = measures(
        sqale_index=60, cognitive_complexity=40, duplicated_lines_density=15
    )
    scores = score_checkpoint((finding,), observed, {}, {}, config, catalog())
    params = config.dimensions[Dimension.MAINTAINABILITY]
    # Independent hand calculation: one severity-3 mark plus three metric marks.
    marks = [3 * (1 - (3 / 5) ** params.kappa), 3 / 2, 3 / 3, 3 / 4]
    weights = [2, 3, 4, 5]
    terms = sum(
        w * params.squale_lambda ** (-mark)
        for mark, w in zip(marks, weights, strict=True)
    )
    expected = (
        -100 / 3 * math.log(terms / sum(weights)) / math.log(params.squale_lambda)
    )
    assert scores.maintainability == pytest.approx(expected)
    arithmetic_mean = (
        sum(w * m for m, w in zip(marks, weights, strict=True)) / sum(weights) * 100 / 3
    )
    assert scores.maintainability < arithmetic_mean
    assert scores.correctness == scores.security == scores.efficiency == 100


def test_issue_count_does_not_change_outer_practice_weights() -> None:
    config = ScoringConfig(dimensions={d: DimensionConfig(rho=0) for d in Dimension})
    finding = issue(IssueType.CODE_SMELL)
    one = score_checkpoint((finding,), measures(), {}, {}, config, catalog())
    many = tuple(replace(finding, key=str(i)) for i in range(100))
    repeated = score_checkpoint(many, measures(), {}, {}, config, catalog())
    assert repeated.maintainability == pytest.approx(one.maintainability)
    assert one.maintainability < 100


def test_four_evidence_sources_each_change_final_maintainability() -> None:
    finding = issue(IssueType.CODE_SMELL)
    metrics = {
        "sqale_index": 60.0,
        "cognitive_complexity": 20.0,
        "duplicated_lines_density": 5.0,
    }
    full = score_checkpoint(
        (finding,), measures(**metrics), {}, {}, ScoringConfig(), catalog()
    )
    no_issues = score_checkpoint(
        (), measures(**metrics), {}, {}, ScoringConfig(), catalog()
    )
    assert full.maintainability < no_issues.maintainability < 100
    for metric in metrics:
        ablated = score_checkpoint(
            (finding,),
            measures(**{**metrics, metric: 0}),
            {},
            {},
            ScoringConfig(),
            catalog(),
        )
        assert full.maintainability < ablated.maintainability


def test_debt_and_complexity_are_densities_but_duplication_is_percentage() -> None:
    original = measures(
        ncloc=100, sqale_index=6, cognitive_complexity=2, duplicated_lines_density=5
    )
    scaled = measures(
        ncloc=10000,
        sqale_index=600,
        cognitive_complexity=200,
        duplicated_lines_density=5,
    )
    assert score_checkpoint(
        (), original, {}, {}, ScoringConfig(), catalog()
    ) == score_checkpoint((), scaled, {}, {}, ScoringConfig(), catalog())


@pytest.mark.parametrize("metric", ["sqale_index", "cognitive_complexity"])
def test_smaller_repository_has_worse_mark_for_same_absolute_metric(
    metric: str,
) -> None:
    small = score_checkpoint(
        (), measures(ncloc=100, **{metric: 20}), {}, {}, ScoringConfig(), catalog()
    )
    large = score_checkpoint(
        (), measures(ncloc=1000, **{metric: 20}), {}, {}, ScoringConfig(), catalog()
    )
    assert small.maintainability < large.maintainability


def test_poor_duplication_outweighs_three_good_practices_nonlinearly() -> None:
    config = ScoringConfig()
    result = score_checkpoint(
        (), measures(duplicated_lines_density=100), {}, {}, config, catalog()
    )
    duplication_mark = 3 / (1 + (100 / config.duplication.tau) ** config.duplication.k)
    arithmetic_mean = (3 + 3 + 3 + duplication_mark) / 4 * 100 / 3
    assert result.maintainability < arithmetic_mean - 25


def test_efficiency_high_blocker_and_removed_classification() -> None:
    config = ScoringConfig()
    rules = catalog(CleanCodeAttribute.EFFICIENT)
    high = replace(
        issue(), impacts=(IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),)
    )
    blocker = replace(
        high, impacts=(IssueImpact(Dimension.CORRECTNESS, Severity.BLOCKER),)
    )
    clean = score_checkpoint((), measures(), {}, {}, config, rules)
    high_scores = score_checkpoint((high,), measures(), {}, {}, config, rules)
    blocker_scores = score_checkpoint((blocker,), measures(), {}, {}, config, rules)
    logical_scores = score_checkpoint((high,), measures(), {}, {}, config, catalog())
    assert blocker_scores.efficiency < high_scores.efficiency < clean.efficiency == 100
    assert logical_scores.correctness == high_scores.correctness < 100
    assert logical_scores.efficiency == 100


def test_complete_checkpoint_uses_modern_impacts_rules_metrics_and_context() -> None:
    first = replace(
        issue(),
        impacts=(
            IssueImpact(Dimension.MAINTAINABILITY, Severity.MAJOR),
            IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),
        ),
    )
    second = replace(
        issue(),
        key="security",
        rule="python:security",
        impacts=(IssueImpact(Dimension.SECURITY, Severity.BLOCKER),),
    )
    rules = RuleCatalog(
        {
            first.rule: SonarRuleMetadata(first.rule, CleanCodeAttribute.EFFICIENT),
            second.rule: SonarRuleMetadata(second.rule, CleanCodeAttribute.TRUSTWORTHY),
        }
    )
    config = ScoringConfig()
    observed = measures(
        ncloc=300, sqale_index=120, cognitive_complexity=45, duplicated_lines_density=8
    )
    centrality = {Path("a.py"): 0.8}
    churn = {Path("a.py"): 0.6}
    result = score_checkpoint(
        (first, second), observed, centrality, churn, config, rules
    )
    for dimension in Dimension:
        assert 0 <= getattr(result, dimension.value.lower()) < 100
    assert result.unmapped_rules == ()
    without_security = score_checkpoint(
        (first,), observed, centrality, churn, config, rules
    )
    assert without_security.security == 100
    assert without_security.correctness == result.correctness
    assert without_security.efficiency == result.efficiency
    assert without_security.maintainability == result.maintainability
    without_maintenance = replace(
        first, impacts=(IssueImpact(Dimension.CORRECTNESS, Severity.CRITICAL),)
    )
    fewer_impacts = score_checkpoint(
        (without_maintenance, second), observed, centrality, churn, config, rules
    )
    assert fewer_impacts.maintainability > result.maintainability
    assert fewer_impacts.correctness == result.correctness
    assert fewer_impacts.security == result.security
    assert fewer_impacts.efficiency == result.efficiency


@pytest.mark.parametrize("dimension", list(Dimension))
def test_each_dimension_uses_severity_frequency_centrality_and_churn(
    dimension: Dimension,
) -> None:
    quality = Dimension.CORRECTNESS if dimension == Dimension.EFFICIENCY else dimension
    finding = replace(issue(), impacts=(IssueImpact(quality, Severity.MAJOR),))
    rules = catalog(CleanCodeAttribute.EFFICIENT)
    config = ScoringConfig()
    baseline = getattr(
        score_checkpoint((finding,), measures(), {}, {}, config, rules),
        dimension.value.lower(),
    )
    higher_severity = replace(
        finding, impacts=(IssueImpact(quality, Severity.CRITICAL),)
    )
    scenarios = [
        ((higher_severity,), {}, {}),
        (tuple(replace(finding, key=str(i)) for i in range(30)), {}, {}),
        ((finding,), {Path("a.py"): 1.0}, {}),
        ((finding,), {}, {Path("a.py"): 1.0}),
    ]
    for findings, centrality, churn in scenarios:
        result = score_checkpoint(
            findings, measures(), centrality, churn, config, rules
        )
        assert getattr(result, dimension.value.lower()) < baseline

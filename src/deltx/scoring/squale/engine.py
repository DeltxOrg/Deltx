"""Pure issue mapping, contextual risk and four-dimension aggregation."""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from deltx.common.exceptions import ConfigurationError
from deltx.scoring.config import ScoringConfig
from deltx.scoring.models import (
    CleanCodeAttribute,
    Dimension,
    IssueImpact,
    QualityScores,
    RuleCatalog,
    Severity,
    SonarIssue,
    SonarMeasures,
    SonarRuleMetadata,
)
from deltx.scoring.squale.formulas import (
    aggregate,
    bounded_log,
    density,
    dynamic_weight,
    individual_mark,
    metric_mark,
)


def map_issue(
    issue: SonarIssue, rule: SonarRuleMetadata, config: ScoringConfig
) -> tuple[IssueImpact, ...]:
    """Preserve software qualities and add EFFICIENT once at maximum severity."""
    if rule.key != issue.rule:
        raise ConfigurationError(
            f"rule metadata does not match issue rule {issue.rule}"
        )
    severities: dict[Dimension, Severity] = {}
    for impact in issue.impacts:
        if impact.dimension == Dimension.EFFICIENCY:
            raise ConfigurationError("Sonar has no EFFICIENCY software quality")
        previous = severities.get(impact.dimension)
        if previous is None or impact.severity.weight > previous.weight:
            severities[impact.dimension] = impact.severity
    if not issue.impacts:
        dimension = config.fallback_mapping.get(issue.issue_type)
        if dimension is not None:
            severities[dimension] = issue.severity
    if rule.clean_code_attribute == CleanCodeAttribute.EFFICIENT:
        severities[Dimension.EFFICIENCY] = (
            max((i.severity for i in issue.impacts), key=lambda s: s.weight)
            if issue.impacts
            else issue.severity
        )
    return tuple(IssueImpact(d, severities[d]) for d in Dimension if d in severities)


def score_checkpoint(
    issues: tuple[SonarIssue, ...],
    measures: SonarMeasures,
    centrality: Mapping[Path, float],
    churn: Mapping[Path, float],
    config: ScoringConfig,
    rule_catalog: RuleCatalog,
) -> QualityScores:
    """Score current state; context must contain only present/past information."""
    rule_catalog.require_efficiency_coverage()
    counts = Counter(issue.rule for issue in issues)
    marks: dict[Dimension, list[tuple[float, float]]] = {d: [] for d in Dimension}
    unmapped: set[str] = set()
    for issue in issues:
        mapping = map_issue(issue, rule_catalog.get(issue.rule), config)
        if not mapping:
            unmapped.add(issue.rule)
        coefficients = next(
            (
                override.coefficients
                for override in config.rule_overrides
                if override.rule == issue.rule
            ),
            {},
        )
        if set(coefficients) - {impact.dimension for impact in mapping}:
            raise ConfigurationError(
                f"rule coefficients for {issue.rule} cannot add dimensions absent "
                "from Sonar impacts/EFFICIENT metadata"
            )
        for impact in mapping:
            dimension = impact.dimension
            params = config.dimensions[dimension]
            weight = dynamic_weight(
                config.severity_values[impact.severity],
                bounded_log(density(counts[issue.rule], measures.ncloc)),
                centrality.get(issue.file, 0.0) if issue.file else 0.0,
                churn.get(issue.file, 0.0) if issue.file else 0.0,
                coefficients.get(dimension, 1.0),
                params,
            )
            marks[dimension].append(
                (individual_mark(weight, params), params.issue_omega)
            )
    # Reduce all maintenance issues to one 0..3 practice mark before mixing
    # with the three project metrics. Issue count cannot change practice weights.
    maintainability = config.dimensions[Dimension.MAINTAINABILITY]
    issue_mark = (
        aggregate(marks[Dimension.MAINTAINABILITY], maintainability.squale_lambda)
        * 3
        / 100
    )
    marks[Dimension.MAINTAINABILITY] = [
        (issue_mark, config.maintainability_issue_omega)
    ]
    synthetic = [
        (density(measures.sqale_index, measures.ncloc), config.debt),
        (density(measures.cognitive_complexity, measures.ncloc), config.complexity),
        (measures.duplicated_lines_density, config.duplication),
    ]
    for value, metric_config in synthetic:
        marks[Dimension.MAINTAINABILITY].append(
            (metric_mark(value, metric_config), metric_config.omega)
        )
    scores = {
        d: aggregate(marks[d], config.dimensions[d].squale_lambda) for d in Dimension
    }
    return QualityScores(
        scores[Dimension.MAINTAINABILITY],
        scores[Dimension.CORRECTNESS],
        scores[Dimension.SECURITY],
        scores[Dimension.EFFICIENCY],
        tuple(sorted(unmapped)),
    )

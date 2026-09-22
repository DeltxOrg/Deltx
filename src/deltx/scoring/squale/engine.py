"""Pure issue mapping, contextual risk and four-dimension aggregation."""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from deltx.scoring.config import ScoringConfig
from deltx.scoring.models import Dimension, QualityScores, SonarIssue, SonarMeasures
from deltx.scoring.squale.formulas import (
    aggregate,
    bounded_log,
    density,
    dynamic_weight,
    individual_mark,
    metric_mark,
)


def map_issue(issue: SonarIssue, config: ScoringConfig) -> dict[Dimension, float]:
    """Explicit Python rule overrides win; no message-keyword inference."""
    for override in config.rule_overrides:
        if override.rule == issue.rule:
            return dict(override.coefficients)
    dimension = config.fallback_mapping.get(issue.issue_type)
    return {} if dimension is None else {dimension: 1.0}


def score_checkpoint(
    issues: tuple[SonarIssue, ...],
    measures: SonarMeasures,
    centrality: Mapping[Path, float],
    churn: Mapping[Path, float],
    config: ScoringConfig,
) -> QualityScores:
    """Score current state; context must contain only present/past information."""
    counts = Counter(issue.rule for issue in issues)
    marks: dict[Dimension, list[tuple[float, float]]] = {d: [] for d in Dimension}
    unmapped: set[str] = set()
    for issue in issues:
        mapping = map_issue(issue, config)
        if not mapping:
            unmapped.add(issue.rule)
        for dimension, coefficient in mapping.items():
            params = config.dimensions[dimension]
            weight = dynamic_weight(
                config.severity_values[issue.severity],
                bounded_log(density(counts[issue.rule], measures.ncloc)),
                centrality.get(issue.file, 0.0) if issue.file else 0.0,
                churn.get(issue.file, 0.0) if issue.file else 0.0,
                coefficient,
                params,
            )
            marks[dimension].append(
                (individual_mark(weight, params), params.issue_omega)
            )
    synthetic = [
        (density(measures.cognitive_complexity, measures.ncloc), config.complexity),
        (measures.duplicated_lines_density, config.duplication),
    ]
    if measures.sqale_index is not None:
        synthetic.append((density(measures.sqale_index, measures.ncloc), config.debt))
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

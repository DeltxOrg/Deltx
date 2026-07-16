"""Dynamic issue weighting using severity, frequency, centrality, and churn.

Computes the per-issue weight ``w_i`` using the formula:

    w_i = S_i · (1 + α · ln(1 + f_i)) · (1 + β · C_i) · (1 + γ · K_i)

where:
- ``S_i``: numeric severity score (1–10)
- ``f_i``: local frequency (count of same rule in the same file)
- ``C_i``: PageRank centrality of the file, in [0, 1]
- ``K_i``: file churn (lines changed over recent history)
- ``α, β, γ``: tunable hyperparameters

Log-scaling on frequency prevents a swarm of identical trivial warnings from
dominating. The multiplicative centrality and churn terms make the same defect
count for more in a hot, central module than in an isolated utility file.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable

from deltx.scoring.iso_mapping import classify
from deltx.scoring.models import Hyperparams, SonarIssue, WeightedIssue

logger = logging.getLogger(__name__)


def weight_issue(
    issue: SonarIssue,
    severity_score: float,
    freq: int,
    centrality: float,
    churn: float,
    hp: Hyperparams,
) -> float:
    """Compute the dynamic weight for a single issue.

    Args:
        issue: The SonarQube issue (used for logging only).
        severity_score: Numeric severity (1–10).
        freq: Local frequency — count of the same rule in the same file.
        centrality: PageRank centrality of the issue's file, in [0, 1].
        churn: Lines changed in the issue's file over recent history.
        hp: Tunable hyperparameters.

    Returns:
        The computed weight ``w_i ≥ 0``.

    Raises:
        ValueError: If any numeric input is negative or NaN.
    """
    # Validate inputs.
    if math.isnan(severity_score) or math.isnan(centrality) or math.isnan(churn):
        msg = (
            f"NaN input: severity={severity_score}, "
            f"centrality={centrality}, churn={churn}"
        )
        raise ValueError(msg)
    if severity_score < 0 or centrality < 0 or churn < 0 or freq < 0:
        msg = (
            f"Negative input: severity={severity_score}, freq={freq}, "
            f"centrality={centrality}, churn={churn}"
        )
        raise ValueError(msg)

    freq_factor = 1.0 + hp.alpha * math.log(1.0 + freq)
    centrality_factor = 1.0 + hp.beta * centrality
    churn_factor = 1.0 + hp.gamma * churn

    weight = severity_score * freq_factor * centrality_factor * churn_factor

    return weight


def weight_all(
    issues: list[SonarIssue],
    centrality_fn: Callable[[str], float],
    churn_fn: Callable[[str], float],
    hp: Hyperparams | None = None,
) -> list[WeightedIssue]:
    """Classify and weight all issues.

    Computes per-rule-per-file frequency internally, then applies the dynamic
    weighting formula to each issue.

    Args:
        issues: Raw SonarQube issues.
        centrality_fn: ``file_path -> centrality ∈ [0, 1]``.
        churn_fn: ``file_path -> churn (lines changed)``.
        hp: Hyperparameters. Uses defaults if None.

    Returns:
        List of weighted issues with dimension classification.
    """
    if hp is None:
        hp = Hyperparams()

    # Compute per-rule-per-file frequency.
    freq_counter: Counter[tuple[str, str]] = Counter()
    for issue in issues:
        key = (issue.component, issue.rule)
        freq_counter[key] += 1

    weighted: list[WeightedIssue] = []
    for issue in issues:
        dimension, severity_score = classify(issue)

        freq = freq_counter[(issue.component, issue.rule)]
        centrality = centrality_fn(issue.component)
        churn = churn_fn(issue.component)

        w = weight_issue(issue, severity_score, freq, centrality, churn, hp)

        weighted.append(WeightedIssue(
            issue=issue,
            dimension=dimension,
            severity_score=severity_score,
            weight=w,
            frequency=freq,
            centrality=centrality,
            churn=churn,
        ))

    logger.info("Weighted %d issues (hp: α=%.2f, β=%.2f, γ=%.2f)",
                len(weighted), hp.alpha, hp.beta, hp.gamma)
    return weighted

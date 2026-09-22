"""Pure calculations using the variable definitions in the research schema."""

import math
from collections.abc import Sequence

from deltx.common.exceptions import ScoringError
from deltx.scoring.config import DimensionConfig, MetricConfig


def bounded_log(value: float) -> float:
    """F' or CH' = log(1+x)/(1+log(1+x)), for nonnegative finite x."""
    if not math.isfinite(value) or value < 0:
        raise ScoringError("log normalization requires finite nonnegative input")
    transformed = math.log1p(value)
    return transformed / (1 + transformed)


def density(count: float, ncloc: float) -> float:
    """Occurrences per KLOC; a zero-LOC checkpoint uses a one-line denominator."""
    return count / max(ncloc, 1) * 1000


def dynamic_weight(
    severity: float,
    frequency: float,
    centrality: float,
    churn: float,
    coefficient: float,
    config: DimensionConfig,
) -> float:
    """W = M*S*[1 + rho*(alpha*F' + beta*C' + gamma*CH')]."""
    if (
        not all(
            math.isfinite(x) and 0 <= x <= 1
            for x in (frequency, centrality, churn, coefficient)
        )
        or not math.isfinite(severity)
        or not 0 < severity <= 5
    ):
        raise ScoringError("invalid severity or normalized issue context")
    return (
        coefficient
        * severity
        * (
            1
            + config.rho
            * (
                config.alpha * frequency
                + config.beta * centrality
                + config.gamma * churn
            )
        )
    )


def individual_mark(weight: float, config: DimensionConfig) -> float:
    """IM = 3*(1 - clip(W/[5*(1+rho)], 0, 1)**kappa)."""
    if not math.isfinite(weight) or weight < 0:
        raise ScoringError("issue weight must be finite and nonnegative")
    risk = min(1.0, max(0.0, weight / (5 * (1 + config.rho))))
    return 3 * (1 - math.pow(risk, config.kappa))


def metric_mark(value: float, config: MetricConfig) -> float:
    """3/[1+(x/tau)**k], evaluated as a stable logistic in log space."""
    if not math.isfinite(value) or value < 0:
        raise ScoringError("synthetic metrics must be finite and nonnegative")
    if value == 0:
        return 3.0
    exponent = config.k * (math.log(value) - math.log(config.tau))
    if exponent >= 0:
        inverse = math.exp(-exponent)
        return 3 * inverse / (1 + inverse)
    return 3 / (1 + math.exp(exponent))


def aggregate(marks: Sequence[tuple[float, float]], squale_lambda: float) -> float:
    """100/3 * -ln(sum(omega*lambda**-IM)/sum(omega))/ln(lambda).

    Log-sum-exp avoids underflow/overflow, and expm1/log1p retain accuracy
    for lambda approaching one. Empty dimensions score 100.
    """
    if not math.isfinite(squale_lambda) or squale_lambda <= 1:
        raise ScoringError("SQUALE lambda must be finite and > 1")
    if not marks:
        return 100.0
    if any(
        not math.isfinite(m) or not 0 <= m <= 3 or not math.isfinite(w) or w <= 0
        for m, w in marks
    ):
        raise ScoringError("SQUALE requires marks in [0,3] and positive finite omega")
    minimum = min(m for m, _ in marks)
    maximum_weight = max(w for _, w in marks)
    log_base = math.log(squale_lambda)
    if log_base < 0.1:
        denominator = math.fsum(w / maximum_weight for _, w in marks)
        adjustment = (
            math.fsum(
                (w / maximum_weight) * math.expm1(-(m - minimum) * log_base)
                for m, w in marks
            )
            / denominator
        )
        log_mean = math.log1p(adjustment)
    else:

        def log_sum(values: list[float]) -> float:
            maximum = max(values)
            return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))

        log_mean = log_sum(
            [math.log(w) - (m - minimum) * log_base for m, w in marks]
        ) - log_sum([math.log(w) for _, w in marks])
    mark = minimum - log_mean / log_base
    return min(100.0, max(0.0, 100 / 3 * mark))

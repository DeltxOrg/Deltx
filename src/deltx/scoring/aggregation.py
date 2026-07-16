"""Squale exponential aggregation for system-level quality scores.

Rolls per-module dimension scores up to a single system-level score per
dimension using the non-linear exponential penalty from the Squale model:

    Score_d = −λ · log_λ( (1/N) · Σ λ^(−s_m / 100) )

This is dominated by the worst module: a single critical defect in a core
routing module collapses the global score. The sensitivity parameter λ
controls the degree of pessimism:

- λ → 1: approaches arithmetic mean (defeats the purpose)
- λ → ∞: approaches pure minimum (throws away all other signal)
- λ ≈ 30: a balanced default that sits meaningfully below the mean
  whenever any module is critical

The implementation uses log base λ via change-of-base:
    log_λ(x) = ln(x) / ln(λ)
"""

from __future__ import annotations

import logging
import math

from deltx.scoring.models import IsoDimension

logger = logging.getLogger(__name__)


def squale_aggregate(module_scores: list[float], lam: float = 30.0) -> float:
    """Compute the Squale exponential aggregate of module scores.

    Args:
        module_scores: Per-module scores in [0, 100].
        lam: Sensitivity parameter (λ). Must be > 1.

    Returns:
        Aggregated system score in [0, 100], dominated by the worst module.

    Raises:
        ValueError: If ``module_scores`` is empty or ``lam ≤ 1``.
    """
    if not module_scores:
        raise ValueError("Cannot aggregate empty module scores")
    if lam <= 1.0:
        raise ValueError(f"λ must be > 1, got {lam}")

    n = len(module_scores)

    if n == 1:
        return module_scores[0]

    log_lam = math.log(lam)

    # Compute λ^(−s_m / 100) for each module.
    # To prevent overflow with large λ: compute in log-space.
    # λ^(−s/100) = exp(−s/100 · ln(λ))
    exp_terms = []
    for s in module_scores:
        # Clamp s to [0, 100] for numerical safety.
        s_clamped = max(0.0, min(100.0, s))
        exp_term = math.exp(-s_clamped / 100.0 * log_lam)
        exp_terms.append(exp_term)

    mean_exp = sum(exp_terms) / n

    # Invert: agg = −λ · log_λ(mean_exp)
    # But we need: agg = −100 · log_λ(mean_exp) / log_λ(λ)
    # Simplification: agg = −100 · ln(mean_exp) / ln(λ)
    # Wait — let's re-derive from the spec formula:
    #   Score = −λ · log_λ( mean(λ^(−s/100)) )
    # This doesn't dimensionally make sense as a [0,100] score.
    # The correct Squale formulation for [0,100] scores is:
    #   Score = −100 · ln(mean(λ^(−s/100))) / ln(λ)

    if mean_exp <= 0:
        return 0.0  # pragma: no cover — numerically impossible for valid inputs

    agg = -100.0 * math.log(mean_exp) / log_lam

    # Clamp to [0, 100] for safety.
    return max(0.0, min(100.0, agg))


def system_scores(
    per_module: dict[IsoDimension, list[float]],
    lam: float = 30.0,
) -> dict[IsoDimension, float]:
    """Compute system-level Squale scores for all four ISO dimensions.

    Args:
        per_module: Mapping from dimension to a list of per-module scores.
        lam: Sensitivity parameter for the Squale aggregation.

    Returns:
        Mapping from each dimension to its system-level aggregated score.
        Dimensions with no modules default to 100.0 (perfect).
    """
    result: dict[IsoDimension, float] = {}
    for dim in IsoDimension:
        scores = per_module.get(dim, [])
        if not scores:
            result[dim] = 100.0
        else:
            result[dim] = squale_aggregate(scores, lam=lam)

    logger.info(
        "System scores (λ=%.1f): %s",
        lam,
        {d.value: f"{s:.1f}" for d, s in result.items()},
    )
    return result

"""ISO/IEC 25010 dimension mapping for SonarQube rules.

Routes every SonarQube issue into one of the four target quality dimensions
(maintainability, correctness, security, efficiency) based on issue type and
optional rule-key overrides.

The mapping strategy is:

1. **Type-based default**: BUG→correctness, VULNERABILITY/SECURITY_HOTSPOT→security,
   CODE_SMELL→maintainability.
2. **Rule-key override**: Specific rules whose SonarQube type is CODE_SMELL but
   whose semantic is efficiency-related (e.g. inefficient loops) are overridden
   into the efficiency dimension.
3. **Fallback**: Any unmapped type falls into maintainability with a warning log.

Severity is mapped to a numeric score: BLOCKER=10, CRITICAL=7, MAJOR=4, MINOR=2, INFO=1.
"""

from __future__ import annotations

import logging

from deltx.scoring.models import IsoDimension, SonarIssue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity → numeric score
# ---------------------------------------------------------------------------

SEVERITY_SCORES: dict[str, float] = {
    "BLOCKER": 10.0,
    "CRITICAL": 7.0,
    "MAJOR": 4.0,
    "MINOR": 2.0,
    "INFO": 1.0,
}


# ---------------------------------------------------------------------------
# Issue type → default dimension
# ---------------------------------------------------------------------------

_TYPE_TO_DIMENSION: dict[str, IsoDimension] = {
    "BUG": IsoDimension.CORRECTNESS,
    "VULNERABILITY": IsoDimension.SECURITY,
    "SECURITY_HOTSPOT": IsoDimension.SECURITY,
    "CODE_SMELL": IsoDimension.MAINTAINABILITY,
}


# ---------------------------------------------------------------------------
# Rule-key overrides (type-based default → actual dimension)
# ---------------------------------------------------------------------------

# Rules that SonarQube classifies as CODE_SMELL but are semantically about
# performance/efficiency. This list covers the most common Python-plugin
# rules; extend as needed.
_EFFICIENCY_RULE_OVERRIDES: set[str] = {
    # Unnecessary collection copies / comprehension waste
    "python:S5765",      # Unnecessary comprehension
    "python:S3516",      # Redundant assignment
    "python:S930",       # Function complexity (indirect efficiency)
    # Generic patterns that multiple language plugins may flag
    "python:InefficiencyRule",
    "python:UnnecessaryCollectionCopy",
    # Common community rules (SonarPython plugin)
    "python:S5717",      # Unused loop variable (wasted iteration)
    "python:S1481",      # Unused local variable
    "python:S1764",      # Identical expressions on both sides of operator
    "python:S3776",      # Cognitive complexity too high (efficiency proxy)
}

# Additional overrides for correctness (BUG-like CODE_SMELLs).
_CORRECTNESS_RULE_OVERRIDES: set[str] = {
    "python:S5607",      # Unreachable code
    "python:S1763",      # All paths return before reaching code
    "python:S3923",      # All branches identical
    "python:S1871",      # Identical branches
}


def classify(issue: SonarIssue) -> tuple[IsoDimension, float]:
    """Classify a SonarQube issue into an ISO dimension and numeric severity.

    Args:
        issue: A normalized SonarQube issue.

    Returns:
        Tuple of (ISO dimension, numeric severity score).
    """
    severity_score = severity_to_score(issue.severity)

    # Check rule-key overrides first (most specific).
    if issue.rule in _EFFICIENCY_RULE_OVERRIDES:
        return IsoDimension.EFFICIENCY, severity_score

    if issue.rule in _CORRECTNESS_RULE_OVERRIDES:
        return IsoDimension.CORRECTNESS, severity_score

    # Fall back to type-based mapping.
    dimension = _TYPE_TO_DIMENSION.get(issue.type)
    if dimension is not None:
        return dimension, severity_score

    # Unknown type — fall back to maintainability with a warning.
    logger.warning(
        "Unmapped issue type %r for rule %r — defaulting to maintainability",
        issue.type,
        issue.rule,
    )
    return IsoDimension.MAINTAINABILITY, severity_score


def severity_to_score(severity: str) -> float:
    """Map a SonarQube severity string to a numeric score.

    Args:
        severity: One of BLOCKER, CRITICAL, MAJOR, MINOR, INFO.

    Returns:
        Numeric severity score (1–10).
    """
    score = SEVERITY_SCORES.get(severity.upper())
    if score is None:
        logger.warning("Unknown severity %r — defaulting to INFO (1.0)", severity)
        return 1.0
    return score


def get_dimension_for_type(issue_type: str) -> IsoDimension | None:
    """Look up the default dimension for a SonarQube issue type.

    Returns None if the type is not in the standard mapping.
    """
    return _TYPE_TO_DIMENSION.get(issue_type)

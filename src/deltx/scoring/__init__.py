"""Squale quality scoring module.

Public surface of Stage 3. Translates raw SonarQube rule violations into four
standardized ISO/IEC 25010 scores on a 0–100 scale using dynamic issue weighting
and Squale-inspired exponential penalty aggregation.

Exports:
    - Data models: :class:`SonarIssue`, :class:`WeightedIssue`,
      :class:`DimensionScore`, :class:`CommitQualityVector`, :class:`Hyperparams`
    - Enums: :class:`IsoDimension`
    - Core API: :func:`score_commit`
    - Client: :class:`SonarClient`
    - Normalizer: :class:`Normalizer`
"""

from __future__ import annotations

from deltx.scoring.models import (
    CommitQualityVector,
    DimensionScore,
    Hyperparams,
    IsoDimension,
    SonarIssue,
    SonarMeasures,
    WeightedIssue,
)

__all__ = [
    "CommitQualityVector",
    "DimensionScore",
    "Hyperparams",
    "IsoDimension",
    "Normalizer",
    "SonarClient",
    "SonarIssue",
    "SonarMeasures",
    "WeightedIssue",
    "score_commit",
]

# Heavy imports are lazy to avoid pulling in networkx/requests when only
# importing data models.
def __getattr__(name: str) -> object:
    """Import heavy classes on first access (PEP 562)."""
    if name == "SonarClient":
        from deltx.scoring.sonar_client import SonarClient
        return SonarClient
    if name == "Normalizer":
        from deltx.scoring.scoring import Normalizer
        return Normalizer
    if name == "score_commit":
        from deltx.scoring.pipeline import score_commit
        return score_commit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy exports in ``dir()``."""
    return sorted(__all__)

"""Dimension penalty accumulation, LOC normalization, and Z-score scoring.

Aggregates weighted issue weights per ISO dimension, normalizes by active LOC,
applies Z-score standardization against persisted training statistics, and
inverts to a bounded 0–100 module score where 100 = perfect (no issues).

The normalizer's μ/σ must be fit on training data **once** and frozen at
inference — refitting per-commit leaks information and makes scores
non-comparable across the time series (which then poisons PatchTST).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deltx.common.exceptions import NormalizerError
from deltx.scoring.models import (
    DimensionScore,
    DimensionStats,
    IsoDimension,
    NormalizerStats,
    WeightedIssue,
)

logger = logging.getLogger(__name__)


def accumulate(weighted_issues: list[WeightedIssue]) -> dict[IsoDimension, float]:
    """Sum weighted issue weights per ISO dimension.

    Args:
        weighted_issues: List of weighted issues with dimension assignments.

    Returns:
        Mapping from each dimension to its raw penalty sum.
        All four dimensions are present; dimensions with no issues get 0.0.
    """
    penalties: dict[IsoDimension, float] = {dim: 0.0 for dim in IsoDimension}
    for wi in weighted_issues:
        penalties[wi.dimension] += wi.weight
    return penalties


def density(raw_penalty: float, loc_active: int) -> float:
    """Normalize a raw penalty by active lines of code.

    Args:
        raw_penalty: Sum of issue weights for a dimension.
        loc_active: Non-comment lines of code in the module.

    Returns:
        Penalty density (penalty / LOC). Returns 0.0 if LOC is zero
        (a file with no code has no defects).
    """
    if loc_active <= 0:
        return 0.0
    return raw_penalty / loc_active


class Normalizer:
    """Z-score normalizer with min-max clipping and score inversion.

    The normalizer must be ``fit()`` on training data before ``transform()``
    can be called. Fitted statistics are persisted as JSON and loaded
    read-only at inference time.

    The ``fingerprint`` field in the persisted stats ties the normalizer
    to a specific training run, preventing scores computed under different
    μ/σ from accidentally entering the same time series.
    """

    def __init__(self) -> None:
        self._stats: NormalizerStats | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the normalizer has been fitted with training data."""
        return self._stats is not None

    @property
    def fingerprint(self) -> str:
        """Training fingerprint, or empty string if not fitted."""
        if self._stats is None:
            return ""
        return self._stats.fingerprint

    def fit(self, densities_by_dim: dict[IsoDimension, list[float]]) -> None:
        """Fit the normalizer on training density data.

        Computes and stores per-dimension mean, std, min, and max.

        Args:
            densities_by_dim: Mapping from dimension to a list of density
                values observed across the training set.

        Raises:
            NormalizerError: If any dimension has fewer than 2 observations.
        """
        dimensions: dict[str, DimensionStats] = {}
        all_values: list[list[float]] = []

        for dim in IsoDimension:
            values = densities_by_dim.get(dim, [])
            if len(values) < 2:
                raise NormalizerError(
                    f"Dimension {dim.value} needs >= 2 observations, "
                    f"got {len(values)}"
                )
            all_values.append(sorted(values))

            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance ** 0.5
            if std == 0:
                std = 1.0  # Prevent division by zero; constant dimension.

            dimensions[dim.value] = DimensionStats(
                mean=mean,
                std=std,
                min_val=min(values),
                max_val=max(values),
            )

        fingerprint = NormalizerStats.compute_fingerprint(all_values)
        self._stats = NormalizerStats(dimensions=dimensions, fingerprint=fingerprint)

        logger.info("Normalizer fitted (fingerprint=%s)", fingerprint)

    def transform(self, penalty_density: float, dimension: IsoDimension) -> float:
        """Transform a penalty density into a 0–100 score.

        Steps:
            1. Z-score: ``z = (P - μ) / σ``
            2. Min-max clip to [0, 1] using training extremes
            3. Invert: ``score = 100 · (1 - clipped)``

        Args:
            penalty_density: Raw penalty / LOC for this dimension.
            dimension: Which ISO dimension to normalize against.

        Returns:
            Score in [0, 100] where 100 = no issues.

        Raises:
            NormalizerError: If the normalizer has not been fitted.
        """
        if self._stats is None:
            raise NormalizerError("Normalizer has not been fitted — call fit() first")

        dim_key = dimension.value
        stats = self._stats.dimensions.get(dim_key)
        if stats is None:
            raise NormalizerError(f"No training statistics for dimension {dim_key}")

        # Z-score standardization.
        z = (penalty_density - stats.mean) / stats.std

        # Min-max normalization of the Z-score to [0, 1].
        z_min = (stats.min_val - stats.mean) / stats.std
        z_max = (stats.max_val - stats.mean) / stats.std

        if z_max == z_min:
            clipped = 0.0
        else:
            normalized = (z - z_min) / (z_max - z_min)
            clipped = max(0.0, min(1.0, normalized))

        # Invert so high penalty → low score.
        score = 100.0 * (1.0 - clipped)
        return score

    def save(self, path: Path) -> None:
        """Persist fitted statistics to a JSON file.

        Raises:
            NormalizerError: If the normalizer has not been fitted.
        """
        if self._stats is None:
            raise NormalizerError("Cannot save unfitted normalizer")
        self._stats.save(path)
        logger.info("Normalizer saved to %s", path)

    def load(self, path: Path, expected_fingerprint: str | None = None) -> None:
        """Load fitted statistics from a JSON file.

        Args:
            path: Path to the normalizer JSON file.
            expected_fingerprint: If provided, the loaded fingerprint must
                match; otherwise ``NormalizerError`` is raised.

        Raises:
            NormalizerError: If the file is missing, malformed, or the
                fingerprint does not match.
        """
        if not path.exists():
            raise NormalizerError(f"Normalizer file not found: {path}")

        try:
            self._stats = NormalizerStats.load(path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise NormalizerError(f"Malformed normalizer file {path}: {exc}") from exc

        if expected_fingerprint is not None:
            if self._stats.fingerprint != expected_fingerprint:
                raise NormalizerError(
                    f"Mismatch: expected {expected_fingerprint!r}, "
                    f"got {self._stats.fingerprint!r}. Scores computed under "
                    f"different μ/σ cannot be mixed in the same time series."
                )

        logger.info(
            "Normalizer loaded from %s (fingerprint=%s)",
            path,
            self._stats.fingerprint,
        )


def module_score(
    weighted_issues: list[WeightedIssue],
    loc: int,
    normalizer: Normalizer,
) -> DimensionScore:
    """Compute per-dimension scores for a single module (file).

    Args:
        weighted_issues: Issues belonging to this module, already weighted.
        loc: Non-comment lines of code in the module.
        normalizer: A fitted :class:`Normalizer`.

    Returns:
        A :class:`DimensionScore` with all four dimension scores.
    """
    penalties = accumulate(weighted_issues)

    scores: dict[IsoDimension, float] = {}
    for dim in IsoDimension:
        d = density(penalties[dim], loc)
        scores[dim] = normalizer.transform(d, dim)

    return DimensionScore.from_dim_dict(scores)

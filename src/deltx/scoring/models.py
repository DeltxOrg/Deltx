"""Pydantic data models for the Squale quality scoring module.

Defines the data contracts flowing through the scoring pipeline:

    SonarQube API → SonarIssue → WeightedIssue → DimensionScore → CommitQualityVector

All models use Pydantic v2 with strict validation. The output contract
(``CommitQualityVector``) is frozen: its four field names match the
``CommitDataVector`` in ``deltx.common.models`` and must never be renamed
without a reviewed migration.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# ISO/IEC 25010 dimension enum
# ---------------------------------------------------------------------------


class IsoDimension(str, enum.Enum):
    """The four target ISO/IEC 25010 quality dimensions."""

    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    SECURITY = "security"
    EFFICIENCY = "efficiency"


# ---------------------------------------------------------------------------
# SonarQube ingestion models
# ---------------------------------------------------------------------------


class SonarIssue(BaseModel):
    """A single issue record normalized from the SonarQube Web API.

    Attributes:
        rule: SonarQube rule key, e.g. ``python:S1234``.
        severity: One of BLOCKER, CRITICAL, MAJOR, MINOR, INFO.
        type: Issue type — BUG, VULNERABILITY, CODE_SMELL, SECURITY_HOTSPOT.
        component: File path relative to the project root.
        line: Source line number (0 if not applicable).
        effort: Remediation effort string, e.g. ``"15min"``.
        message: Human-readable issue description.
    """

    rule: str
    severity: str
    type: str
    component: str
    line: int = 0
    effort: str = "0min"
    message: str = ""


class SonarMeasures(BaseModel):
    """Project-level measures fetched from SonarQube ``/api/measures/component``.

    Attributes:
        ncloc: Non-comment lines of code.
        complexity: Cyclomatic complexity.
        cognitive_complexity: Cognitive complexity.
        duplicated_lines_density: Percentage of duplicated lines.
    """

    ncloc: int = 0
    complexity: int = 0
    cognitive_complexity: int = 0
    duplicated_lines_density: float = 0.0


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class Hyperparams:
    """Tunable hyperparameters for the weighting and aggregation formulas.

    Attributes:
        alpha: Log-frequency sensitivity. Higher → more weight to repeated rules.
        beta: Centrality sensitivity. Higher → more weight to topologically central files.
        gamma: Churn sensitivity. Higher → more weight to frequently changed files.
        lam: Squale exponential aggregation sensitivity (λ).
              Higher → closer to pure minimum; lower → closer to arithmetic mean.
    """

    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.3
    lam: float = 30.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to a plain dict for JSON persistence."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lam": self.lam,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Hyperparams:
        """Deserialize from a plain dict."""
        return cls(
            alpha=data["alpha"],
            beta=data["beta"],
            gamma=data["gamma"],
            lam=data["lam"],
        )

    @classmethod
    def from_json(cls, path: Path) -> Hyperparams:
        """Load from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Path) -> None:
        """Persist to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Weighted issue
# ---------------------------------------------------------------------------


class WeightedIssue(BaseModel):
    """A SonarQube issue enriched with its computed dynamic weight.

    Attributes:
        issue: The original SonarQube issue.
        dimension: The ISO dimension this issue maps to.
        severity_score: Numeric severity (1–10).
        weight: Computed dynamic weight ``w_i``.
        frequency: Count of same rule in the same file/module.
        centrality: PageRank centrality of the issue's file.
        churn: Lines changed in the issue's file over recent history.
    """

    issue: SonarIssue
    dimension: IsoDimension
    severity_score: float
    weight: float = Field(..., ge=0)
    frequency: int = Field(..., ge=0)
    centrality: float = Field(..., ge=0, le=1)
    churn: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Dimension scores
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Per-module scores for all four ISO/IEC 25010 dimensions.

    Each score is in [0, 100] where 100 = no issues detected.
    """

    maintainability: float = Field(100.0, ge=0, le=100)
    correctness: float = Field(100.0, ge=0, le=100)
    security: float = Field(100.0, ge=0, le=100)
    efficiency: float = Field(100.0, ge=0, le=100)

    def to_dict(self) -> dict[IsoDimension, float]:
        """Return scores keyed by ``IsoDimension``."""
        return {
            IsoDimension.MAINTAINABILITY: self.maintainability,
            IsoDimension.CORRECTNESS: self.correctness,
            IsoDimension.SECURITY: self.security,
            IsoDimension.EFFICIENCY: self.efficiency,
        }

    @classmethod
    def from_dim_dict(cls, scores: dict[IsoDimension, float]) -> DimensionScore:
        """Construct from a ``{IsoDimension: score}`` mapping."""
        return cls(
            maintainability=scores.get(IsoDimension.MAINTAINABILITY, 100.0),
            correctness=scores.get(IsoDimension.CORRECTNESS, 100.0),
            security=scores.get(IsoDimension.SECURITY, 100.0),
            efficiency=scores.get(IsoDimension.EFFICIENCY, 100.0),
        )


# ---------------------------------------------------------------------------
# Output contract: the four quality scores
# ---------------------------------------------------------------------------


class CommitQualityVector(BaseModel):
    """The four ISO/IEC 25010 quality scores produced by Stage 3.

    Field names are **architecture-frozen**: they match the corresponding
    fields in ``deltx.common.models.CommitDataVector`` exactly. Renaming
    any field here without updating the 15-D vector breaks PatchTST training.
    """

    score_maintainability: float = Field(..., ge=0, le=100)
    score_correctness: float = Field(..., ge=0, le=100)
    score_security: float = Field(..., ge=0, le=100)
    score_efficiency: float = Field(..., ge=0, le=100)

    @classmethod
    def field_names(cls) -> list[str]:
        """Return the four score field names in canonical order."""
        return [
            "score_maintainability",
            "score_correctness",
            "score_security",
            "score_efficiency",
        ]

    def to_dict(self) -> dict[str, float]:
        """Return scores as a plain dict keyed by field name."""
        return {
            "score_maintainability": self.score_maintainability,
            "score_correctness": self.score_correctness,
            "score_security": self.score_security,
            "score_efficiency": self.score_efficiency,
        }


# ---------------------------------------------------------------------------
# Normalizer statistics (persisted to JSON)
# ---------------------------------------------------------------------------


@dataclass
class DimensionStats:
    """Persisted training statistics for a single ISO dimension."""

    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class NormalizerStats:
    """Per-dimension training statistics for the Z-score normalizer.

    Persisted as JSON and loaded read-only at inference time. The
    ``fingerprint`` field ties the stats to a specific training run so
    scores computed under different μ/σ cannot accidentally enter the
    same time series.
    """

    dimensions: dict[str, DimensionStats] = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "fingerprint": self.fingerprint,
            "dimensions": {
                dim: {
                    "mean": stats.mean,
                    "std": stats.std,
                    "min_val": stats.min_val,
                    "max_val": stats.max_val,
                }
                for dim, stats in self.dimensions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NormalizerStats:
        """Deserialize from a JSON-compatible dict."""
        dims_raw = data.get("dimensions", {})
        if not isinstance(dims_raw, dict):
            dims_raw = {}
        dimensions = {}
        for dim, stats_dict in dims_raw.items():
            if isinstance(stats_dict, dict):
                dimensions[dim] = DimensionStats(
                    mean=float(stats_dict.get("mean", 0.0)),
                    std=float(stats_dict.get("std", 1.0)),
                    min_val=float(stats_dict.get("min_val", 0.0)),
                    max_val=float(stats_dict.get("max_val", 1.0)),
                )
        fp = data.get("fingerprint", "")
        return cls(dimensions=dimensions, fingerprint=str(fp))

    def save(self, path: Path) -> None:
        """Persist to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> NormalizerStats:
        """Load from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def compute_fingerprint(density_values: list[list[float]]) -> str:
        """Compute a deterministic fingerprint from training density data."""
        raw = json.dumps(density_values, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Validation: CommitQualityVector field alignment
# ---------------------------------------------------------------------------


class _QualityVectorContract(BaseModel):
    """Internal validator ensuring CommitQualityVector stays aligned.

    This model is never instantiated directly — it exists so the contract
    test can import and verify field-name alignment without reaching
    into private implementation details.
    """

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_fields(self) -> _QualityVectorContract:
        """Validate that CommitQualityVector exposes the expected fields."""
        expected = {"score_maintainability", "score_correctness",
                    "score_security", "score_efficiency"}
        actual = set(CommitQualityVector.model_fields.keys())
        if actual != expected:
            msg = (
                f"CommitQualityVector field drift detected. "
                f"Expected {sorted(expected)}, got {sorted(actual)}"
            )
            raise ValueError(msg)
        return self

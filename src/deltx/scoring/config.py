"""Versioned research BASELINE, deliberately not empirically optimized."""

import math
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from deltx.scoring.models import Dimension, IssueType, Severity


class FrozenConfig(BaseModel):
    """Reject typos and non-finite parameters at the configuration boundary."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True
    )

    @field_validator("*", mode="after")
    @classmethod
    def freeze_mapping(cls, value: object) -> object:
        """Frozen models must also protect nested dictionaries from mutation."""
        return MappingProxyType(value) if isinstance(value, dict) else value

    @field_serializer("*", check_fields=False)
    def serialize_mapping(self, value: object) -> object:
        """Keep immutable mappings compatible with reproducibility JSON."""
        return dict(value) if isinstance(value, Mapping) else value


class DimensionConfig(FrozenConfig):
    """Context mixture, risk shape, aggregation base and issue weight."""

    alpha: float = Field(default=1 / 3, ge=0)
    beta: float = Field(default=1 / 3, ge=0)
    gamma: float = Field(default=1 / 3, ge=0)
    rho: float = Field(default=1, ge=0)
    kappa: float = Field(default=1, gt=0)
    squale_lambda: float = Field(default=9, gt=1)
    issue_omega: float = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_mixture(self) -> "DimensionConfig":
        if not math.isclose(
            self.alpha + self.beta + self.gamma, 1, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("alpha + beta + gamma must equal 1")
        if not math.isfinite(5 * (1 + self.rho)):
            raise ValueError("rho overflows maximum dynamic risk")
        return self


class MetricConfig(FrozenConfig):
    """Provisional half-mark threshold tau, shape k, and aggregation weight."""

    tau: float = Field(gt=0)
    k: float = Field(default=1, gt=0)
    omega: float = Field(default=1, gt=0)


class RuleMapping(FrozenConfig):
    """Explicit rule override; multiple dimensions may have M in (0, 1]."""

    rule: str
    coefficients: Mapping[Dimension, float]

    @model_validator(mode="after")
    def validate_coefficients(self) -> "RuleMapping":
        if not self.coefficients or any(
            not math.isfinite(v) or not 0 < v <= 1 for v in self.coefficients.values()
        ):
            raise ValueError("mapping coefficients must be finite and in (0, 1]")
        return self


class ScoringConfig(FrozenConfig):
    """All research hyperparameters, serializable as a single JSON document.

    The initial curated efficiency signal is Python S2190 (unbounded recursion,
    wasting CPU/stack resources). It also retains its correctness influence.
    Broader performance coverage requires reviewed rule overrides.
    """

    version: str = "PYTHON_RESEARCH_BASELINE_V1"
    dimensions: Mapping[Dimension, DimensionConfig] = Field(
        default_factory=lambda: {d: DimensionConfig() for d in Dimension}
    )
    severity_values: Mapping[Severity, float] = Field(
        default_factory=lambda: {s: float(i) for i, s in enumerate(Severity, 1)}
    )
    fallback_mapping: Mapping[IssueType, Dimension] = Field(
        default_factory=lambda: {
            IssueType.BUG: Dimension.CORRECTNESS,
            IssueType.VULNERABILITY: Dimension.SECURITY,
            IssueType.SECURITY_HOTSPOT: Dimension.SECURITY,
            IssueType.CODE_SMELL: Dimension.MAINTAINABILITY,
        }
    )
    rule_overrides: tuple[RuleMapping, ...] = Field(
        default_factory=lambda: (
            RuleMapping(
                rule="python:S2190",
                coefficients={
                    Dimension.CORRECTNESS: 1.0,
                    Dimension.EFFICIENCY: 1.0,
                },
            ),
        )
    )
    debt: MetricConfig = Field(default_factory=lambda: MetricConfig(tau=1000))
    complexity: MetricConfig = Field(default_factory=lambda: MetricConfig(tau=100))
    duplication: MetricConfig = Field(default_factory=lambda: MetricConfig(tau=5))
    churn_horizon: int = Field(default=50, gt=0)
    churn_decay: float = Field(default=0.9, gt=0, le=1)
    pagerank_alpha: float = Field(default=0.85, gt=0, lt=1)
    pagerank_tolerance: float = Field(default=1e-12, gt=0)
    pagerank_max_iterations: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def validate_complete(self) -> "ScoringConfig":
        if set(self.dimensions) != set(Dimension):
            raise ValueError("all four dimensions must be configured")
        values = [self.severity_values.get(s, math.nan) for s in Severity]
        if (
            any(not math.isfinite(v) or not 0 < v <= 5 for v in values)
            or values != sorted(set(values))
            or values[-1] != 5
        ):
            raise ValueError("severity values must increase strictly up to BLOCKER=5")
        if len({r.rule for r in self.rule_overrides}) != len(self.rule_overrides):
            raise ValueError("duplicate rule overrides")
        return self

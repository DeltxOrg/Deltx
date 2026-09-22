"""Typed, immutable inputs and outputs of the scoring domain."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from deltx.common.exceptions import ConfigurationError


class Severity(StrEnum):
    """Five ordinal buckets; MQR LOW/MEDIUM/HIGH normalize to MINOR/MAJOR/CRITICAL."""

    INFO = "INFO"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"
    BLOCKER = "BLOCKER"

    @property
    def weight(self) -> int:
        """Ordinal severity, independent of spelling and external API vocabulary."""
        return tuple(Severity).index(self) + 1


class Dimension(StrEnum):
    """The four Deltx target dimensions."""

    MAINTAINABILITY = "MAINTAINABILITY"
    CORRECTNESS = "CORRECTNESS"
    SECURITY = "SECURITY"
    EFFICIENCY = "EFFICIENCY"


class IssueType(StrEnum):
    """Reliable Sonar classifications; unknown types remain observable."""

    BUG = "BUG"
    VULNERABILITY = "VULNERABILITY"
    SECURITY_HOTSPOT = "SECURITY_HOTSPOT"
    CODE_SMELL = "CODE_SMELL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class IssueImpact:
    """One software quality's impact, normalized at the Sonar API boundary."""

    dimension: Dimension
    severity: Severity


class CleanCodeAttribute(StrEnum):
    """Sonar rule attributes; only EFFICIENT identifies static efficiency rules."""

    FORMATTED = "FORMATTED"
    CONVENTIONAL = "CONVENTIONAL"
    IDENTIFIABLE = "IDENTIFIABLE"
    CLEAR = "CLEAR"
    LOGICAL = "LOGICAL"
    COMPLETE = "COMPLETE"
    EFFICIENT = "EFFICIENT"
    FOCUSED = "FOCUSED"
    DISTINCT = "DISTINCT"
    MODULAR = "MODULAR"
    TESTED = "TESTED"
    LAWFUL = "LAWFUL"
    TRUSTWORTHY = "TRUSTWORTHY"
    RESPECTFUL = "RESPECTFUL"


@dataclass(frozen=True)
class SonarRuleMetadata:
    """Active Python rule metadata, parsed once outside the scoring engine."""

    key: str
    clean_code_attribute: CleanCodeAttribute
    impacts: tuple[IssueImpact, ...] = ()


@dataclass(frozen=True)
class RuleCatalog:
    """Immutable lookup of the active Python rules for one profile identity."""

    rules: Mapping[str, SonarRuleMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(key != rule.key for key, rule in self.rules.items()):
            raise ConfigurationError("rule catalog keys do not match rule metadata")
        object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))

    def get(self, rule_key: str) -> SonarRuleMetadata:
        """Require metadata; never guess whether an unknown rule is EFFICIENT."""
        try:
            return self.rules[rule_key]
        except KeyError as exc:
            raise ConfigurationError(
                f"missing active Python rule metadata for {rule_key}"
            ) from exc

    @property
    def efficiency_rule_keys(self) -> tuple[str, ...]:
        """Active rules whose Clean Code attribute is exactly EFFICIENT."""
        return tuple(
            sorted(
                key
                for key, rule in self.rules.items()
                if rule.clean_code_attribute == CleanCodeAttribute.EFFICIENT
            )
        )

    def require_efficiency_coverage(self) -> None:
        """Absence of applicable rules cannot be reported as a perfect score."""
        if not self.efficiency_rule_keys:
            raise ConfigurationError(
                "The active SonarQube Python quality profile contains no "
                "EFFICIENT rules; score_efficiency cannot be computed."
            )


@dataclass(frozen=True)
class SonarIssue:
    """One current issue, with a repository-relative file if available."""

    key: str
    rule: str
    severity: Severity
    issue_type: IssueType
    file: Path | None
    impacts: tuple[IssueImpact, ...] = ()


class SonarMeasures(BaseModel):
    """Raw project-state measures; all three maintainability practices are required."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")
    ncloc: float = Field(ge=0)
    cognitive_complexity: float = Field(ge=0)
    duplicated_lines_density: float = Field(ge=0, le=100)
    sqale_index: float = Field(ge=0)
    sqale_debt_ratio: float | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class Analysis:
    """A completed Sonar checkpoint, with reproducibility evidence."""

    issues: tuple[SonarIssue, ...]
    measures: SonarMeasures
    sonar_version: str
    profile: str
    scanner_version: str
    analysis_id: str = ""
    scanner_image_id: str = ""
    issue_model: str = "MQR_EFFICIENT_V2"
    rule_catalog: RuleCatalog = field(kw_only=True)


@dataclass(frozen=True)
class QualityScores:
    """Four scores and unmapped rule keys for orchestration to report."""

    maintainability: float
    correctness: float
    security: float
    efficiency: float
    unmapped_rules: tuple[str, ...] = ()

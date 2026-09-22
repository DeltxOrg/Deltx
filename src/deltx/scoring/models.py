"""Typed, immutable inputs and outputs of the scoring domain."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    """Five ordinal buckets; MQR LOW/MEDIUM/HIGH normalize to MINOR/MAJOR/CRITICAL."""

    INFO = "INFO"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"
    BLOCKER = "BLOCKER"


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
    """Raw project-state measures; absent optional debt remains absent."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")
    ncloc: float = Field(ge=0)
    cognitive_complexity: float = Field(ge=0)
    duplicated_lines_density: float = Field(ge=0, le=100)
    sqale_index: float | None = Field(default=None, ge=0)
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
    issue_model: str = "MQR_PREFERRED_V1"


@dataclass(frozen=True)
class QualityScores:
    """Four scores and unmapped rule keys for orchestration to report."""

    maintainability: float
    correctness: float
    security: float
    efficiency: float
    unmapped_rules: tuple[str, ...] = ()

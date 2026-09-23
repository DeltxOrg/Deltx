"""Shared Pydantic data models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommitDataVector(BaseModel):
    """The 15 numeric ML features; CSV repository/commit identifiers are separate."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    score_maintainability: float = Field(ge=0, le=100)
    score_correctness: float = Field(ge=0, le=100)
    score_security: float = Field(ge=0, le=100)
    score_efficiency: float = Field(ge=0, le=100)
    ai_confidence_pct: float = Field(ge=0, le=100)
    loc_added: int = Field(ge=0)
    loc_deleted: int = Field(ge=0)
    files_modified_count: int = Field(ge=0)
    avg_pagerank_centrality: float = Field(ge=0, le=1)
    density_blocker_issues: float = Field(ge=0)
    density_critical_issues: float = Field(ge=0)
    density_major_issues: float = Field(ge=0)
    density_minor_issues: float = Field(ge=0)
    cognitive_complexity: float = Field(ge=0)
    duplication_density: float = Field(ge=0, le=100)

    @classmethod
    def quality_score_fields(cls) -> list[str]:
        """Return the names of the four ISO/IEC 25010 quality score fields."""
        return [
            "score_maintainability",
            "score_correctness",
            "score_security",
            "score_efficiency",
        ]

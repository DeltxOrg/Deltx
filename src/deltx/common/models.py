"""Shared Pydantic data models."""

from __future__ import annotations

from pydantic import BaseModel


class CommitDataVector(BaseModel):
    """The canonical 15-D vector representing a commit in the ML dataset."""

    commit_size: float
    file_count: float
    complexity_delta: float
    churn_rate: float
    ai_confidence_pct: float
    score_maintainability: float
    score_correctness: float
    score_security: float
    score_efficiency: float
    author_experience: float
    time_since_last_commit: float
    test_coverage_delta: float
    dependency_count_delta: float
    documentation_ratio: float
    coupling_score: float

    @classmethod
    def quality_score_fields(cls) -> list[str]:
        """Return the names of the four ISO/IEC 25010 quality score fields."""
        return [
            "score_maintainability",
            "score_correctness",
            "score_security",
            "score_efficiency",
        ]

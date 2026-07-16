"""Shared Pydantic data models.

Defines the canonical 15-dimensional commit data vector that all pipeline stages
bind against. Individual modules (detection, scoring, etc.) populate their own
subset of fields; the full vector is assembled by Stage 1 (extraction).
"""

from pydantic import BaseModel, Field


class CommitDataVector(BaseModel):
    """Canonical 15-dimensional feature vector for a single commit.

    This is the single source of truth for the vector schema that PatchTST
    consumes. Every field name and type defined here is an architecture
    contract — renaming, retyping, or removing a field breaks downstream
    training and must be explicitly reviewed.

    Fields are grouped by the producing stage:

    - Stage 2 (Detection): ``ai_confidence_pct``
    - Stage 3 (Scoring): ``score_maintainability``, ``score_correctness``,
      ``score_security``, ``score_efficiency``
    - Stage 1 (Extraction): all remaining metrics

    All scores are floats in [0, 100]. Metric fields are non-negative floats
    whose ranges depend on the specific metric.
    """

    # --- Stage 1: Extraction metrics ---
    commit_size: float = Field(
        ..., ge=0, description="Total lines changed in the commit"
    )
    file_count: float = Field(
        ..., ge=0, description="Number of files modified in the commit"
    )
    complexity_delta: float = Field(
        ..., description="Change in cyclomatic complexity vs. parent commit"
    )
    churn_rate: float = Field(
        ..., ge=0, description="Lines changed / total LOC in modified files"
    )

    # --- Stage 2: Detection ---
    ai_confidence_pct: float = Field(
        ..., ge=0, le=100,
        description="Probability (0–100) that commit code is AI-authored",
    )

    # --- Stage 3: Scoring (ISO/IEC 25010) ---
    score_maintainability: float = Field(
        ..., ge=0, le=100,
        description="ISO 25010 maintainability score (0–100, 100 = perfect)",
    )
    score_correctness: float = Field(
        ..., ge=0, le=100,
        description="ISO 25010 correctness/reliability score (0–100)",
    )
    score_security: float = Field(
        ..., ge=0, le=100,
        description="ISO 25010 security score (0–100)",
    )
    score_efficiency: float = Field(
        ..., ge=0, le=100,
        description="ISO 25010 performance-efficiency score (0–100)",
    )

    # --- Stage 1: Additional extraction metrics ---
    author_experience: float = Field(
        ..., ge=0, description="Author's prior commit count in this repo"
    )
    time_since_last_commit: float = Field(
        ..., ge=0, description="Hours since the author's previous commit"
    )
    test_coverage_delta: float = Field(
        ..., description="Change in test coverage percentage vs. parent"
    )
    dependency_count_delta: float = Field(
        ..., description="Change in number of external dependencies"
    )
    documentation_ratio: float = Field(
        ..., ge=0, description="Docstring + comment lines / total lines"
    )
    coupling_score: float = Field(
        ..., ge=0, description="Inter-module coupling metric (fan-out)"
    )

    # --- Schema introspection ---

    @classmethod
    def quality_score_fields(cls) -> list[str]:
        """Return the four ISO/IEC 25010 quality score field names.

        These are the fields produced by the scoring module (Stage 3) and
        consumed as PatchTST prediction targets.
        """
        return [
            "score_maintainability",
            "score_correctness",
            "score_security",
            "score_efficiency",
        ]

    @classmethod
    def all_field_names(cls) -> list[str]:
        """Return all 15 field names in canonical order."""
        return list(cls.model_fields.keys())

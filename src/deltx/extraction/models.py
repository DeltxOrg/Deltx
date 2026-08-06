"""Data model for one row of the extraction output.

Each :class:`CommitRow` is one commit's contribution to the Parquet table that
feeds the downstream time-series stage. Rows are emitted for *every* commit on
the branch, including commits that touched no Python files, so the series has no
gaps.
"""

import json
import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deltx.common.constants import AI_CONFIDENCE_MAX, AI_CONFIDENCE_MIN


class CommitRow(BaseModel):
    """One commit's row in the extraction output table.

    Attributes:
        repo_url: The repository this row came from.
        commit_hash: Full 40-character SHA.
        commit_timestamp: Author timestamp, timezone-aware and UTC-normalised.
        commit_author: Author name.
        commit_message: First line of the commit message.
        commit_index: 0-based position in oldest-first traversal (0 = oldest).
        files_changed_py: Count of ``.py`` files added or modified in the commit.
        total_loc_scored: Total lines of code across the files actually scored.
        ai_confidence_pct: LOC-weighted AI confidence in ``[0, 100]``, or ``NaN``
            when the commit had nothing to score (no changed ``.py`` files, or
            none of them yielded scoreable code). ``NaN`` is deliberate: a commit
            with no evidence must not masquerade as a confident ``0.0`` human
            signal in the series.
        file_scores_json: JSON object ``{path: {"score", "loc"}}`` over the
            scored files, for traceability of the weighted average. ``score`` is
            the file's ``P(AI)`` in ``[0, 1]`` — the quantity the aggregate
            weights — not the percentage.
    """

    model_config = ConfigDict(frozen=True)

    repo_url: str
    commit_hash: str
    commit_timestamp: datetime
    commit_author: str
    commit_message: str
    commit_index: int = Field(ge=0)
    files_changed_py: int = Field(ge=0)
    total_loc_scored: int = Field(ge=0)
    ai_confidence_pct: float
    file_scores_json: str = "{}"

    @field_validator("ai_confidence_pct")
    @classmethod
    def _in_range_or_nan(cls, value: float) -> float:
        """Accept a percentage in range, or ``NaN`` for an unscored commit."""
        if math.isnan(value):
            return value
        if not AI_CONFIDENCE_MIN <= value <= AI_CONFIDENCE_MAX:
            msg = (
                f"ai_confidence_pct must be in "
                f"[{AI_CONFIDENCE_MIN}, {AI_CONFIDENCE_MAX}] or NaN, got {value!r}"
            )
            raise ValueError(msg)
        return value

    @classmethod
    def empty(
        cls,
        *,
        repo_url: str,
        commit_hash: str,
        commit_timestamp: datetime,
        commit_author: str,
        commit_message: str,
        commit_index: int,
    ) -> "CommitRow":
        """Build the row for a commit that changed no scoreable Python files.

        Returns:
            A row with ``files_changed_py`` and ``total_loc_scored`` at zero and
            ``ai_confidence_pct`` at ``NaN``.
        """
        return cls(
            repo_url=repo_url,
            commit_hash=commit_hash,
            commit_timestamp=commit_timestamp,
            commit_author=commit_author,
            commit_message=commit_message,
            commit_index=commit_index,
            files_changed_py=0,
            total_loc_scored=0,
            ai_confidence_pct=math.nan,
            file_scores_json="{}",
        )

    @staticmethod
    def encode_file_scores(file_scores: dict[str, dict[str, float | int]]) -> str:
        """Serialise the per-file score/LOC map to a compact JSON string."""
        return json.dumps(file_scores, separators=(",", ":"), sort_keys=True)

"""Pydantic data models for the AI authorship detection module."""

from datetime import datetime
from enum import IntEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deltx.common.constants import AI_CONFIDENCE_MAX, AI_CONFIDENCE_MIN


class DroidLabel(IntEnum):
    """The four authorship classes DroidDetect predicts.

    Index order comes from the model card, not from the checkpoint: the
    published ``config.json`` carries no ``id2label``, so nothing in the
    artifact pins this mapping. Reading it wrong inverts ``ai_confidence_pct``
    silently, so the order is asserted in the test suite.
    """

    HUMAN_GENERATED = 0
    MACHINE_GENERATED = 1
    MACHINE_REFINED = 2
    MACHINE_GENERATED_ADVERSARIAL = 3


class ClassDistribution(BaseModel):
    """A softmax distribution over the four :class:`DroidLabel` classes."""

    model_config = ConfigDict(frozen=True)

    human_generated: float = Field(ge=0.0, le=1.0)
    machine_generated: float = Field(ge=0.0, le=1.0)
    machine_refined: float = Field(ge=0.0, le=1.0)
    machine_generated_adversarial: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_normalised(self) -> "ClassDistribution":
        """Reject distributions that do not sum to 1 within float tolerance."""
        total = (
            self.human_generated
            + self.machine_generated
            + self.machine_refined
            + self.machine_generated_adversarial
        )
        if abs(total - 1.0) > 1e-4:
            msg = f"class probabilities must sum to 1.0, got {total!r}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_probabilities(cls, probabilities: list[float]) -> "ClassDistribution":
        """Build from a 4-element probability list in :class:`DroidLabel` order.

        Args:
            probabilities: Softmax output ordered by ``DroidLabel`` index.

        Returns:
            The corresponding distribution.

        Raises:
            ValueError: If exactly four probabilities were not supplied.
        """
        if len(probabilities) != len(DroidLabel):
            msg = f"expected {len(DroidLabel)} probabilities, got {len(probabilities)}"
            raise ValueError(msg)
        return cls(
            human_generated=probabilities[DroidLabel.HUMAN_GENERATED],
            machine_generated=probabilities[DroidLabel.MACHINE_GENERATED],
            machine_refined=probabilities[DroidLabel.MACHINE_REFINED],
            machine_generated_adversarial=probabilities[
                DroidLabel.MACHINE_GENERATED_ADVERSARIAL
            ],
        )

    @property
    def p_ai(self) -> float:
        """Probability of any AI involvement: ``1 - P(HUMAN_GENERATED)``.

        Collapsing the three machine classes into their complement of class 0
        turns the noisy 4-way decision into the reliable human-vs-machine one:
        a ``MACHINE_GENERATED`` sample misread as ``MACHINE_REFINED`` still
        scores high, because the confusion stays inside the collapsed group.
        """
        return 1.0 - self.human_generated

    @property
    def predicted_label(self) -> DroidLabel:
        """The argmax class, retained for diagnostics rather than for scoring."""
        ordered = [
            self.human_generated,
            self.machine_generated,
            self.machine_refined,
            self.machine_generated_adversarial,
        ]
        return DroidLabel(ordered.index(max(ordered)))


class FileAnalysisResult(BaseModel):
    """The outcome of scoring one source file."""

    model_config = ConfigDict(frozen=True)

    file_path: Path
    ai_confidence: float = Field(ge=0.0, le=1.0)
    distribution: ClassDistribution | None = None
    lines_of_code: int = Field(ge=0)
    token_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=1, ge=0)
    is_scored: bool = True
    error_message: str | None = None

    @classmethod
    def unscored(
        cls, file_path: Path, reason: str, lines_of_code: int = 0
    ) -> "FileAnalysisResult":
        """Build a result for a file that could not be scored.

        Assumes human authorship when in doubt, and marks the file so commit
        aggregation excludes it rather than averaging in a fabricated 0.0.

        Args:
            file_path: Path of the file that failed.
            reason: Human-readable explanation.
            lines_of_code: Line count, if it was determined before failure.

        Returns:
            An unscored result carrying the reason.
        """
        return cls(
            file_path=file_path,
            ai_confidence=0.0,
            lines_of_code=lines_of_code,
            chunk_count=0,
            is_scored=False,
            error_message=reason,
        )


class CommitAnalysisResult(BaseModel):
    """The commit-level detection score consumed by downstream stages."""

    model_config = ConfigDict(frozen=True)

    commit_hash: str
    timestamp: datetime
    author: str | None = None
    ai_confidence_pct: float = Field(ge=AI_CONFIDENCE_MIN, le=AI_CONFIDENCE_MAX)
    file_results: tuple[FileAnalysisResult, ...] = ()
    files_analyzed: int = Field(default=0, ge=0)
    files_skipped: int = Field(default=0, ge=0)

    @classmethod
    def aggregate(
        cls,
        commit_hash: str,
        timestamp: datetime,
        file_results: list[FileAnalysisResult],
        files_skipped: int = 0,
        author: str | None = None,
    ) -> "CommitAnalysisResult":
        """Combine file scores into one commit score by LOC-weighted average.

        Only scored files contribute. A file with zero lines of code carries no
        weight, so a commit whose scored files are all empty falls back to an
        unweighted mean rather than dividing by zero. A commit with nothing
        scorable at all scores 0.0 — assume human when in doubt.

        Args:
            commit_hash: Identifying hash of the commit.
            timestamp: Commit timestamp.
            file_results: Per-file results, scored and unscored alike.
            files_skipped: Count of files excluded before scoring.
            author: Commit author, if known.

        Returns:
            The aggregated commit result.
        """
        scored = [r for r in file_results if r.is_scored]
        total_weight = sum(r.lines_of_code for r in scored)

        if not scored:
            confidence = 0.0
        elif total_weight == 0:
            confidence = sum(r.ai_confidence for r in scored) / len(scored)
        else:
            confidence = (
                sum(r.ai_confidence * r.lines_of_code for r in scored) / total_weight
            )

        return cls(
            commit_hash=commit_hash,
            timestamp=timestamp,
            author=author,
            ai_confidence_pct=round(confidence * AI_CONFIDENCE_MAX, 4),
            file_results=tuple(file_results),
            files_analyzed=len(scored),
            files_skipped=files_skipped + (len(file_results) - len(scored)),
        )

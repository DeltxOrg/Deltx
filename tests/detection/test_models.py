"""Tests for the detection data models, including commit aggregation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from deltx.detection.models import (
    ClassDistribution,
    CommitAnalysisResult,
    DroidLabel,
    FileAnalysisResult,
)

TIMESTAMP = datetime(2026, 7, 30, tzinfo=UTC)


def make_distribution(human: float) -> ClassDistribution:
    """A distribution with the remaining mass split across the machine classes."""
    rest = (1.0 - human) / 3
    return ClassDistribution(
        human_generated=human,
        machine_generated=rest,
        machine_refined=rest,
        machine_generated_adversarial=rest,
    )


def scored(path: str, confidence: float, loc: int) -> FileAnalysisResult:
    """A scored file result."""
    return FileAnalysisResult(
        file_path=Path(path),
        ai_confidence=confidence,
        distribution=make_distribution(1.0 - confidence),
        lines_of_code=loc,
    )


class TestDroidLabel:
    def test_index_order_is_pinned(self) -> None:
        """Verified against DroidCollection ground truth; nothing in the
        checkpoint pins this, so a regression here silently inverts scores."""
        assert DroidLabel.HUMAN_GENERATED == 0
        assert DroidLabel.MACHINE_GENERATED == 1
        assert DroidLabel.MACHINE_REFINED == 2
        assert DroidLabel.MACHINE_GENERATED_ADVERSARIAL == 3
        assert len(DroidLabel) == 4


class TestClassDistribution:
    def test_rejects_unnormalised(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            ClassDistribution(
                human_generated=0.9,
                machine_generated=0.9,
                machine_refined=0.0,
                machine_generated_adversarial=0.0,
            )

    def test_from_probabilities_maps_in_label_order(self) -> None:
        dist = ClassDistribution.from_probabilities([0.1, 0.2, 0.3, 0.4])
        assert dist.human_generated == 0.1
        assert dist.machine_generated == 0.2
        assert dist.machine_refined == 0.3
        assert dist.machine_generated_adversarial == 0.4

    def test_from_probabilities_rejects_wrong_arity(self) -> None:
        with pytest.raises(ValueError, match="expected 4"):
            ClassDistribution.from_probabilities([0.5, 0.5])

    def test_p_ai_collapses_all_three_machine_classes(self) -> None:
        """Confusion among machine classes must not move the score."""
        spread = ClassDistribution.from_probabilities([0.2, 0.3, 0.3, 0.2])
        concentrated = ClassDistribution.from_probabilities([0.2, 0.8, 0.0, 0.0])
        assert spread.p_ai == pytest.approx(0.8)
        assert concentrated.p_ai == pytest.approx(0.8)

    def test_predicted_label(self) -> None:
        dist = ClassDistribution.from_probabilities([0.1, 0.2, 0.6, 0.1])
        assert dist.predicted_label is DroidLabel.MACHINE_REFINED

    def test_is_frozen(self) -> None:
        dist = make_distribution(0.5)
        with pytest.raises(ValidationError):
            dist.human_generated = 0.9


class TestFileAnalysisResult:
    def test_unscored_marks_and_explains(self) -> None:
        result = FileAnalysisResult.unscored(Path("a.txt"), "not a Python file")
        assert result.is_scored is False
        assert result.ai_confidence == 0.0
        assert result.chunk_count == 0
        assert result.error_message == "not a Python file"

    def test_confidence_must_be_a_probability(self) -> None:
        with pytest.raises(ValidationError):
            FileAnalysisResult(
                file_path=Path("a.py"), ai_confidence=1.5, lines_of_code=1
            )


class TestCommitAggregation:
    def test_weights_by_lines_of_code(self) -> None:
        """A large confident file must dominate a small one."""
        result = CommitAnalysisResult.aggregate(
            commit_hash="abc",
            timestamp=TIMESTAMP,
            file_results=[scored("big.py", 1.0, 900), scored("small.py", 0.0, 100)],
        )
        assert result.ai_confidence_pct == pytest.approx(90.0)
        assert result.files_analyzed == 2

    def test_unscored_files_are_excluded_not_averaged_in(self) -> None:
        """An unscored file must not drag the average toward zero."""
        result = CommitAnalysisResult.aggregate(
            commit_hash="abc",
            timestamp=TIMESTAMP,
            file_results=[
                scored("a.py", 1.0, 10),
                FileAnalysisResult.unscored(Path("b.txt"), "not Python"),
            ],
        )
        assert result.ai_confidence_pct == pytest.approx(100.0)
        assert result.files_analyzed == 1
        assert result.files_skipped == 1

    def test_commit_with_nothing_scorable_is_zero(self) -> None:
        """Assume human when in doubt."""
        result = CommitAnalysisResult.aggregate(
            commit_hash="abc",
            timestamp=TIMESTAMP,
            file_results=[FileAnalysisResult.unscored(Path("a.txt"), "not Python")],
        )
        assert result.ai_confidence_pct == 0.0
        assert result.files_analyzed == 0

    def test_empty_commit_is_zero(self) -> None:
        result = CommitAnalysisResult.aggregate("abc", TIMESTAMP, [])
        assert result.ai_confidence_pct == 0.0

    def test_zero_loc_files_fall_back_to_unweighted_mean(self) -> None:
        """Total weight of zero must not divide by zero."""
        result = CommitAnalysisResult.aggregate(
            commit_hash="abc",
            timestamp=TIMESTAMP,
            file_results=[scored("a.py", 1.0, 0), scored("b.py", 0.0, 0)],
        )
        assert result.ai_confidence_pct == pytest.approx(50.0)

    def test_skipped_count_includes_pre_scoring_skips(self) -> None:
        result = CommitAnalysisResult.aggregate(
            commit_hash="abc",
            timestamp=TIMESTAMP,
            file_results=[scored("a.py", 0.5, 10)],
            files_skipped=3,
        )
        assert result.files_skipped == 3

    def test_output_is_a_percentage(self) -> None:
        result = CommitAnalysisResult.aggregate(
            "abc", TIMESTAMP, [scored("a.py", 0.42, 10)]
        )
        assert result.ai_confidence_pct == pytest.approx(42.0)
        assert 0.0 <= result.ai_confidence_pct <= 100.0

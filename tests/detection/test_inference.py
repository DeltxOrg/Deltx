"""Tests for file- and commit-level inference with the detector mocked."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from deltx.common.exceptions import DetectionError
from deltx.detection.detector import ScoredSource
from deltx.detection.inference import (
    AIDetectionInference,
    count_lines_of_code,
    skip_reason,
)
from deltx.detection.models import ClassDistribution

TIMESTAMP = datetime(2026, 7, 30, tzinfo=UTC)
SOURCE = "def f():\n    return 1\n"


class StubDetector:
    """Returns a fixed score, or raises, without loading anything."""

    def __init__(self, p_ai: float = 0.8, fail: bool = False) -> None:
        self.p_ai = p_ai
        self.fail = fail

    def score_source(self, source: str) -> ScoredSource:
        if self.fail:
            msg = "stub failure"
            raise DetectionError(msg)
        return ScoredSource(
            distribution=ClassDistribution(
                human_generated=1.0 - self.p_ai,
                machine_generated=self.p_ai,
            ),
            token_count=10,
            chunk_count=1,
        )


@pytest.fixture
def inference() -> AIDetectionInference:
    return AIDetectionInference(StubDetector())  # type: ignore[arg-type]


class TestSkipReason:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (Path("module.py"), None),
            (Path("notes.md"), "not a Python file"),
            (Path("setup.py"), "packaging or test boilerplate"),
            (Path("conftest.py"), "packaging or test boilerplate"),
            (Path("pkg/__pycache__/mod.py"), "generated directory"),
            (Path("src/deltx/detection/models.py"), None),
        ],
    )
    def test_skip_rules(self, path: Path, expected: str | None) -> None:
        assert skip_reason(path) == expected


class TestCountLinesOfCode:
    def test_excludes_blank_and_comment_only_lines(self) -> None:
        source = "# header\n\ndef f():\n    # inner\n    return 1\n\n"
        assert count_lines_of_code(source) == 2

    def test_empty_source_is_zero(self) -> None:
        assert count_lines_of_code("") == 0

    def test_trailing_comment_on_a_code_line_still_counts(self) -> None:
        assert count_lines_of_code("x = 1  # set x\n") == 1


class TestAnalyzeFile:
    def test_scores_a_python_file(self, inference: AIDetectionInference) -> None:
        result = inference.analyze_file(SOURCE, Path("m.py"))
        assert result.is_scored
        assert result.ai_confidence == pytest.approx(0.8)
        assert result.distribution is not None

    def test_skipped_file_is_unscored(self, inference: AIDetectionInference) -> None:
        result = inference.analyze_file("# notes", Path("notes.md"))
        assert result.is_scored is False
        assert result.error_message == "not a Python file"

    def test_file_with_no_code_lines_is_unscored(
        self, inference: AIDetectionInference
    ) -> None:
        result = inference.analyze_file("# only a comment\n", Path("m.py"))
        assert result.is_scored is False
        assert result.error_message == "no code lines"

    def test_detector_failure_is_contained(self) -> None:
        """analyze_file must never raise: one bad file cannot fail a commit."""
        inference = AIDetectionInference(StubDetector(fail=True))  # type: ignore[arg-type]
        result = inference.analyze_file(SOURCE, Path("m.py"))
        assert result.is_scored is False
        assert "stub failure" in (result.error_message or "")
        assert result.lines_of_code == 2


class TestAnalyzeCommit:
    def test_aggregates_python_files_and_skips_others(
        self, inference: AIDetectionInference
    ) -> None:
        result = inference.analyze_commit(
            files={
                Path("a.py"): SOURCE,
                Path("b.py"): SOURCE,
                Path("README.md"): "# docs",
            },
            commit_hash="a1b2c3d4",
            timestamp=TIMESTAMP,
            author="alice",
        )
        assert result.ai_confidence_pct == pytest.approx(80.0)
        assert result.files_analyzed == 2
        assert result.files_skipped == 1
        assert result.author == "alice"

    def test_commit_of_only_non_python_scores_zero(
        self, inference: AIDetectionInference
    ) -> None:
        result = inference.analyze_commit(
            files={Path("README.md"): "# docs"},
            commit_hash="a1b2c3d4",
            timestamp=TIMESTAMP,
        )
        assert result.ai_confidence_pct == 0.0
        assert result.files_analyzed == 0

    def test_empty_commit_scores_zero(self, inference: AIDetectionInference) -> None:
        result = inference.analyze_commit({}, "abc", TIMESTAMP)
        assert result.ai_confidence_pct == 0.0

    def test_batch_preserves_order(self, inference: AIDetectionInference) -> None:
        commits = [
            {
                "files": {Path("a.py"): SOURCE},
                "commit_hash": f"hash{i}" * 2,
                "timestamp": TIMESTAMP,
            }
            for i in range(3)
        ]
        results = inference.analyze_commit_batch(commits)  # type: ignore[arg-type]
        assert [r.commit_hash for r in results] == [c["commit_hash"] for c in commits]

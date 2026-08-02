"""File- and commit-level inference: the module's public entry point."""

import logging
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from deltx.common.config import DeltxConfig
from deltx.common.constants import (
    PYTHON_SUFFIX,
    SKIPPED_DIR_PARTS,
    SKIPPED_FILENAMES,
)
from deltx.common.exceptions import DeltxError
from deltx.detection.detector import DroidDetector
from deltx.detection.models import CommitAnalysisResult, FileAnalysisResult

logger = logging.getLogger(__name__)


class CommitRecord(TypedDict):
    """One commit's inputs for batch analysis."""

    files: dict[Path, str]
    commit_hash: str
    timestamp: datetime
    author: NotRequired[str | None]


def skip_reason(file_path: Path) -> str | None:
    """Decide whether a path is outside the module's remit.

    Args:
        file_path: Path to test.

    Returns:
        A reason string when the file should be skipped, else ``None``.
    """
    if file_path.suffix != PYTHON_SUFFIX:
        return "not a Python file"
    if file_path.name in SKIPPED_FILENAMES:
        return "packaging or test boilerplate"
    if SKIPPED_DIR_PARTS.intersection(file_path.parts):
        return "generated directory"
    return None


def count_lines_of_code(source: str) -> int:
    """Count non-empty, non-comment-only lines.

    This is the weight used for commit aggregation, so it measures authored
    code rather than file size.

    Args:
        source: Python source text.

    Returns:
        The line count.
    """
    return sum(
        1
        for line in source.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )


class AIDetectionInference:
    """Scores files and commits, producing ``ai_confidence_pct``."""

    def __init__(self, detector: DroidDetector) -> None:
        """Wrap a loaded detector.

        Args:
            detector: The loaded DroidDetect wrapper.
        """
        self.detector = detector

    @classmethod
    def from_config(cls, config: DeltxConfig | None = None) -> "AIDetectionInference":
        """Load the detector and build the inference facade.

        Args:
            config: Active configuration. A default is built when omitted.

        Returns:
            A ready inference pipeline.
        """
        return cls(DroidDetector.from_config(config or DeltxConfig()))

    def analyze_file(self, source_code: str, file_path: Path) -> FileAnalysisResult:
        """Score one file. Never raises.

        A file that cannot be scored is returned unscored rather than as a
        fabricated 0.0, so commit aggregation can exclude it instead of
        diluting the average.

        Args:
            source_code: The file's contents.
            file_path: Its path, used for skip rules and reporting.

        Returns:
            The per-file result.
        """
        reason = skip_reason(file_path)
        if reason is not None:
            return FileAnalysisResult.unscored(file_path, reason)

        lines_of_code = count_lines_of_code(source_code)
        if lines_of_code == 0:
            return FileAnalysisResult.unscored(file_path, "no code lines")

        try:
            scored = self.detector.score_source(source_code)
        except DeltxError as exc:
            logger.warning("could not score %s: %s", file_path, exc)
            return FileAnalysisResult.unscored(file_path, str(exc), lines_of_code)

        return FileAnalysisResult(
            file_path=file_path,
            ai_confidence=scored.distribution.p_ai,
            distribution=scored.distribution,
            lines_of_code=lines_of_code,
            token_count=scored.token_count,
            chunk_count=scored.chunk_count,
        )

    def analyze_commit(
        self,
        files: dict[Path, str],
        commit_hash: str,
        timestamp: datetime,
        author: str | None = None,
    ) -> CommitAnalysisResult:
        """Score every Python file in a commit and aggregate to one score.

        Args:
            files: Mapping of path to file contents for the commit.
            commit_hash: Identifying hash.
            timestamp: Commit timestamp.
            author: Commit author, if known.

        Returns:
            The commit result, carrying ``ai_confidence_pct``.
        """
        results = [
            self.analyze_file(source, path) for path, source in sorted(files.items())
        ]
        return CommitAnalysisResult.aggregate(
            commit_hash=commit_hash,
            timestamp=timestamp,
            file_results=results,
            author=author,
        )

    def analyze_commit_batch(
        self, commits: list[CommitRecord]
    ) -> list[CommitAnalysisResult]:
        """Score a sequence of commits in order.

        Args:
            commits: Commit records to analyse.

        Returns:
            One result per input commit, in the same order.
        """
        results: list[CommitAnalysisResult] = []
        for index, commit in enumerate(commits, start=1):
            logger.info(
                "analysing commit %d/%d (%s)",
                index,
                len(commits),
                commit["commit_hash"][:8],
            )
            results.append(
                self.analyze_commit(
                    files=commit["files"],
                    commit_hash=commit["commit_hash"],
                    timestamp=commit["timestamp"],
                    author=commit.get("author"),
                )
            )
        return results

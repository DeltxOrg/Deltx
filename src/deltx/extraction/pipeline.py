"""Per-commit orchestration: changed files -> scores -> one output row.

This layer is deliberately thin. The heavy lifting — skip rules, chunking,
per-file scoring, and the LOC-weighted aggregation into ``ai_confidence_pct`` —
already lives in :mod:`deltx.detection`. The extractor's only job is to feed a
commit's changed Python files into that pipeline and shape the result into a
:class:`CommitRow`, including the cases the detection module never sees: a commit
that changed no Python files at all.
"""

import logging
import math
from datetime import UTC
from pathlib import Path

from deltx.detection.inference import AIDetectionInference
from deltx.extraction.git_history import CommitMeta, GitRepository
from deltx.extraction.models import CommitRow

logger = logging.getLogger(__name__)


class CommitAiConfidenceExtractor:
    """Turns commits into scored :class:`CommitRow` records."""

    def __init__(
        self,
        inference: AIDetectionInference,
        repository: GitRepository,
        repo_url: str,
    ) -> None:
        """Wire together the detector, the repository, and the source URL.

        Args:
            inference: Loaded detection pipeline (injected so tests can supply a
                fake without downloading the checkpoint).
            repository: Handle over the cloned repository.
            repo_url: URL recorded verbatim on every output row.
        """
        self.inference = inference
        self.repository = repository
        self.repo_url = repo_url

    def build_row(self, commit: CommitMeta, index: int) -> CommitRow:
        """Score one commit and shape it into an output row.

        Args:
            commit: The commit to score.
            index: Its 0-based position in oldest-first traversal.

        Returns:
            The row for this commit — never ``None``; a commit with nothing to
            score still produces a row so the series stays gap-free.
        """
        timestamp = commit.timestamp.astimezone(UTC)
        changed = self.repository.changed_python_files(commit)

        if not changed:
            logger.debug("commit %s changed no .py files", commit.commit_hash[:8])
            return CommitRow.empty(
                repo_url=self.repo_url,
                commit_hash=commit.commit_hash,
                commit_timestamp=timestamp,
                commit_author=commit.author,
                commit_message=commit.message,
                commit_index=index,
            )

        files: dict[Path, str] = {}
        for path in changed:
            content = self.repository.read_file(commit.commit_hash, path)
            if content is None:
                # Binary or undecodable — already warned in read_file.
                continue
            files[Path(path.as_posix())] = content

        result = self.inference.analyze_commit(
            files=files,
            commit_hash=commit.commit_hash,
            timestamp=timestamp,
            author=commit.author,
        )

        scored = [r for r in result.file_results if r.is_scored]
        for unscored in (r for r in result.file_results if not r.is_scored):
            logger.debug(
                "commit %s: not scoring %s (%s)",
                commit.commit_hash[:8],
                unscored.file_path.as_posix(),
                unscored.error_message,
            )

        total_loc = sum(r.lines_of_code for r in scored)
        file_scores: dict[str, dict[str, float | int]] = {
            r.file_path.as_posix(): {
                "score": round(r.ai_confidence, 6),
                "loc": r.lines_of_code,
            }
            for r in scored
        }

        # `result.ai_confidence_pct` is already the LOC-weighted average * 100
        # over the scored files. Fall back to NaN when nothing scored, so a
        # commit with no evidence is never recorded as a confident 0.0 human.
        pct = result.ai_confidence_pct if total_loc > 0 else math.nan

        return CommitRow(
            repo_url=self.repo_url,
            commit_hash=commit.commit_hash,
            commit_timestamp=timestamp,
            commit_author=commit.author,
            commit_message=commit.message,
            commit_index=index,
            files_changed_py=len(changed),
            total_loc_scored=total_loc,
            ai_confidence_pct=pct,
            file_scores_json=CommitRow.encode_file_scores(file_scores),
        )

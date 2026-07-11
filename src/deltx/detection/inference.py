"""Production inference pipeline for AI authorship detection (Stage 2 entry point).

This is the module the rest of Deltx calls. It ties the two halves of the
detection stage together — the :class:`FeatureExtractionPipeline` (raw source →
16-D vector) and the trained :class:`DetectionClassifier` (vector → AI
probability) — and exposes them behind a small commit-oriented API::

    detector = AIDetectionInference.from_config(config)
    result = detector.analyze_commit(commit_files, commit_hash, timestamp)
    result.ai_confidence_pct  # 0–100; becomes index [4] of the 15-D commit vector

**Granularity.** A file is classified to a probability in ``[0, 1]``; a commit's
``ai_confidence_pct`` is the LOC-weighted average of its files, scaled to
``[0, 100]`` — the integration contract's file → commit aggregation.

**"Assume human when in doubt."** A file whose features cannot be trusted — it did
not parse, or a feature family raised — is *not* fed to the classifier, because a
partially-zeroed vector would be classified against signals the model never
trained on. Such a file is returned with ``ai_confidence=0.0`` and
``is_parseable=False`` so it is excluded from the commit average entirely, rather
than dragged toward a spurious "0.0 = human" reading. If *no* file in a commit is
classifiable, the commit scores ``0.0``.

Like the pipeline it wraps, this layer never lets one pathological file abort a
run: :meth:`analyze_file` contains its own failures so a batch of commits always
completes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from deltx.common.config import DeltxConfig
from deltx.detection.classifier import DetectionClassifier
from deltx.detection.models import (
    CommitAnalysisResult,
    FeatureVector,
    FileAnalysisResult,
)
from deltx.detection.pipeline import FeatureExtractionPipeline

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# Files that are Python by name but carry no authorship signal worth scoring:
# packaging boilerplate and pytest configuration are near-identical across
# repositories, so including them would only add noise to the commit average.
_SKIP_FILENAMES: frozenset[str] = frozenset({"setup.py", "conftest.py"})

# Any path component equal to this marks compiled-cache output, never source.
_PYCACHE_DIR = "__pycache__"


class CommitRecord(TypedDict):
    """One commit's inputs for :meth:`AIDetectionInference.analyze_commit_batch`."""

    files: dict[Path, str]
    commit_hash: str
    timestamp: datetime
    author: NotRequired[str | None]


def _is_analyzable(path: Path) -> bool:
    """Return True if ``path`` is a Python source file worth analyzing.

    Excludes non-``.py`` files (``.pyc`` included, by suffix), the packaging and
    test-config files in :data:`_SKIP_FILENAMES`, and anything under a
    ``__pycache__`` directory.
    """
    if path.suffix != ".py":
        return False
    if path.name in _SKIP_FILENAMES:
        return False
    return _PYCACHE_DIR not in path.parts


def _commit_progress() -> Progress:
    """A rich progress bar sized for commit-oriented batch work (current/total)."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )


class AIDetectionInference:
    """Production inference pipeline for AI authorship detection.

    This is the main entry point for the detection module.

    Usage::

        detector = AIDetectionInference.from_config(config)
        result = detector.analyze_commit(commit_files, commit_hash, timestamp)
        print(result.ai_confidence_pct)  # 0-100 scale
    """

    def __init__(
        self,
        pipeline: FeatureExtractionPipeline,
        classifier: DetectionClassifier,
    ) -> None:
        """Initialise with pre-built components.

        Args:
            pipeline: Feature extractor producing 16-D vectors from source.
            classifier: A *fitted* classifier; its ``predict_proba`` supplies the
                per-file AI probability. Construction does not check fitness —
                :meth:`from_config` guarantees it, and callers wiring components by
                hand are expected to pass a trained classifier.
        """
        self.pipeline = pipeline
        self.classifier = classifier

    @classmethod
    def from_config(cls, config: DeltxConfig) -> AIDetectionInference:
        """Build the pipeline and load the trained classifier from config paths.

        Args:
            config: Global configuration; supplies the language-model settings for
                the pipeline and ``classifier_path`` for the saved model.

        Returns:
            A ready-to-use inference pipeline.

        Raises:
            ModelNotLoadedError: If no classifier file exists at
                ``config.classifier_path``.
        """
        pipeline = FeatureExtractionPipeline(config)
        classifier = DetectionClassifier(config)
        # Raises ModelNotLoadedError when the file is absent or malformed.
        classifier.load()
        logger.info(
            "AI detection inference ready (model=%s, classifier=%s)",
            config.model_name,
            config.classifier_path,
        )
        return cls(pipeline, classifier)

    def analyze_file(
        self, source_code: str, file_path: Path
    ) -> FileAnalysisResult:
        """Analyze a single Python file.

        Extracts the feature vector, and — only if extraction was clean — runs the
        classifier to populate ``ai_confidence``. If feature extraction fails (the
        file did not parse, or a feature family raised), the file is treated as
        unclassifiable: the result carries ``is_parseable=False`` and
        ``ai_confidence=0.0`` (assume human when we cannot classify), which also
        excludes it from any commit-level average.

        Args:
            source_code: Raw Python source of one file.
            file_path: Path the source came from (reporting only).

        Returns:
            A :class:`FileAnalysisResult` with ``ai_confidence`` in ``[0, 1]`` for
            a classified file, or ``0.0`` for an unclassifiable one.
        """
        try:
            result = self.pipeline.extract_file_features(source_code, file_path)
        except Exception as exc:  # noqa: BLE001 - one file must not sink the run
            logger.warning("Feature extraction crashed for %s: %s", file_path, exc)
            return self._unclassifiable(file_path, f"extraction error: {exc}")

        # A partial vector (bad parse, or a family that raised) is worse than no
        # classification: the model never saw such rows, so we do not guess.
        if not result.is_parseable or result.error_message is not None:
            logger.debug(
                "Not classifying %s (%s); assuming human",
                file_path,
                result.error_message or "source did not parse",
            )
            return result.model_copy(
                update={"ai_confidence": 0.0, "is_parseable": False}
            )

        confidence = self._classify(result.feature_vector)
        return result.model_copy(update={"ai_confidence": confidence})

    def analyze_commit(
        self,
        files: dict[Path, str],
        commit_hash: str,
        timestamp: datetime,
        author: str | None = None,
    ) -> CommitAnalysisResult:
        """Analyze all modified Python files in a commit.

        Non-Python and boilerplate files (see :func:`_is_analyzable`) are skipped
        before analysis. The commit's ``ai_confidence_pct`` is the LOC-weighted
        average over classifiable files — parseable, with ``lines_of_code > 0`` —
        scaled to ``[0, 100]``. A commit with nothing classifiable scores ``0.0``.

        Args:
            files: ``{file_path: source_code}`` for the files touched by the commit.
            commit_hash: The commit's hash (reporting only).
            timestamp: The commit's timestamp.
            author: Optional commit author, carried onto the result.

        Returns:
            A :class:`CommitAnalysisResult` holding the commit score and every
            file-level result (including skipped-analysis and unclassifiable ones).
        """
        file_results: list[FileAnalysisResult] = []
        skipped = 0
        for file_path, source_code in files.items():
            if not _is_analyzable(file_path):
                logger.debug("Skipping non-analyzable file %s", file_path)
                skipped += 1
                continue
            file_results.append(self.analyze_file(source_code, file_path))

        classifiable = [
            result
            for result in file_results
            if result.is_parseable and result.lines_of_code > 0
        ]
        ai_confidence_pct = (
            CommitAnalysisResult.aggregate(classifiable) if classifiable else 0.0
        )

        logger.info(
            "Commit %s: %d files analyzed, ai_confidence=%.1f%%",
            commit_hash[:8],
            len(file_results),
            ai_confidence_pct,
        )
        return CommitAnalysisResult(
            commit_hash=commit_hash,
            timestamp=timestamp,
            author=author,
            ai_confidence_pct=ai_confidence_pct,
            file_results=file_results,
            total_files_analyzed=len(file_results),
            total_files_skipped=skipped,
        )

    def analyze_commit_batch(
        self,
        commits: list[CommitRecord],
        progress: bool = True,
    ) -> list[CommitAnalysisResult]:
        """Batch-analyze multiple commits, optionally with a progress bar.

        Args:
            commits: Each record carries ``files``, ``commit_hash``, ``timestamp``
                and an optional ``author`` — the arguments of :meth:`analyze_commit`.
            progress: Whether to render a rich progress bar over the commits.

        Returns:
            One :class:`CommitAnalysisResult` per input commit, in the same order.
        """
        results: list[CommitAnalysisResult] = []
        if progress and commits:
            with _commit_progress() as bar:
                task = bar.add_task("Analyzing commits", total=len(commits))
                for commit in commits:
                    results.append(self._analyze_record(commit))
                    bar.advance(task)
        else:
            for commit in commits:
                results.append(self._analyze_record(commit))
        return results

    def _analyze_record(self, commit: CommitRecord) -> CommitAnalysisResult:
        """Unpack one :class:`CommitRecord` and analyze it."""
        return self.analyze_commit(
            files=commit["files"],
            commit_hash=commit["commit_hash"],
            timestamp=commit["timestamp"],
            author=commit.get("author"),
        )

    def _classify(self, feature_vector: FeatureVector) -> float:
        """Run the classifier on one feature vector, returning P(AI) in ``[0, 1]``."""
        matrix = feature_vector.to_array().reshape(1, -1)
        proba = self.classifier.predict_proba(matrix)
        return float(np.asarray(proba, dtype=np.float64).reshape(-1)[0])

    @staticmethod
    def _unclassifiable(file_path: Path, error_message: str) -> FileAnalysisResult:
        """Build a zeroed, unparseable result for a file that yielded nothing."""
        zeros = dict.fromkeys(FeatureVector.feature_names(), 0.0)
        return FileAnalysisResult(
            file_path=file_path,
            feature_vector=FeatureVector(**zeros),
            ai_confidence=0.0,
            lines_of_code=0,
            is_parseable=False,
            error_message=error_message,
        )


__all__ = ["AIDetectionInference", "CommitRecord"]

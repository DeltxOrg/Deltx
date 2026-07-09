"""Feature extraction orchestrator for the AI authorship detection module.

Drives the parser and all three feature families to turn a raw Python source
string into a complete 16-dimensional :class:`FeatureVector`:

===============  ========  ==================================================
Family           Features  Input
===============  ========  ==================================================
Perplexity       F1–F6     raw ``source_code`` (scored by the code LM)
Stylometric      F7–F12    :class:`ParsedSource` (AST + line statistics)
Distribution     F13–F16   :class:`ParsedSource` (lexical token stream)
===============  ========  ==================================================

**Error policy.** A commit may touch hundreds of files, and one pathological
file must never abort the run. So failure is contained at two levels:

* A family that raises contributes zeros for *its* features only; the other two
  families still produce real values, and the reason is recorded on
  ``FileAnalysisResult.error_message``.
* :meth:`FeatureExtractionPipeline.extract_batch` additionally isolates each
  file, so an unexpected failure outside the families still yields a result row.

Note that the perplexity family reads the *raw source*, not the parse: the
language model scores byte sequences and neither knows nor cares whether the
file is valid Python. A file that fails to parse therefore still carries a
meaningful F1–F6 signal, which is exactly the case where authorship evidence is
scarcest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from rich.logging import RichHandler

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import FeatureExtractionError
from deltx.detection.features import (
    DistributionExtractor,
    PerplexityExtractor,
    StylometricExtractor,
)
from deltx.detection.models import FeatureVector, FileAnalysisResult
from deltx.detection.parser import PythonSourceParser

logger = logging.getLogger(__name__)

# Package-root logger; the rich handler is attached here so every `deltx.*`
# module inherits it rather than each configuring its own.
_ROOT_LOGGER_NAME = "deltx"

# The 16 feature names in F1–F16 order, partitioned by the family that owns
# them. FeatureVector is the single source of truth for the ordering.
_ALL_FEATURE_NAMES: tuple[str, ...] = tuple(FeatureVector.feature_names())
_PERPLEXITY_KEYS: tuple[str, ...] = _ALL_FEATURE_NAMES[0:6]
_STYLOMETRIC_KEYS: tuple[str, ...] = _ALL_FEATURE_NAMES[6:12]
_DISTRIBUTION_KEYS: tuple[str, ...] = _ALL_FEATURE_NAMES[12:16]


def _configure_rich_logging() -> None:
    """Attach a rich handler to the ``deltx`` logger, once per process.

    Idempotent: a second call (or a caller that has already configured logging
    themselves) is a no-op, so constructing several pipelines does not stack
    duplicate handlers.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if root.handlers:
        return
    root.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    root.setLevel(logging.INFO)


class FeatureExtractionPipeline:
    """Orchestrates all feature extractors to produce a complete 16-D FeatureVector."""

    def __init__(self, config: DeltxConfig) -> None:
        """Initialise the parser and the three feature extractors.

        The language model backing :class:`PerplexityExtractor` is *not* fetched
        here — it loads lazily on the first file that needs scoring — so building
        a pipeline stays cheap and offline.

        Args:
            config: Global configuration, forwarded to the perplexity extractor
                for model name, cache directory, device, and thresholds.
        """
        _configure_rich_logging()
        self.config = config
        self.parser = PythonSourceParser()
        self.perplexity = PerplexityExtractor(config)
        self.stylometric = StylometricExtractor()
        self.distribution = DistributionExtractor()

    def extract_file_features(
        self, source_code: str, file_path: Path
    ) -> FileAnalysisResult:
        """Extract all 16 features from a single Python file.

        The primary interface. Never raises for a single file: a failed parse
        still yields whatever features survive it, and a failed feature family
        contributes zeros while the others proceed.

        Args:
            source_code: Raw Python source of one file.
            file_path: Path the source came from (reporting only).

        Returns:
            A :class:`FileAnalysisResult` whose ``ai_confidence`` is a ``0.0``
            placeholder — the classifier fills it in downstream. ``is_parseable``
            reflects the parse; ``error_message`` names any family that failed.
        """
        # Nothing to score and nothing to parse. Short-circuiting here also
        # spares an empty file the cost of waking the language model.
        if not source_code or not source_code.strip():
            logger.warning("Empty source for %s; returning zeroed vector", file_path)
            return self._zeroed_result(file_path, "Source file is empty")

        parsed = self.parser.parse(source_code, file_path)
        if not parsed.is_valid:
            logger.warning(
                "%s did not parse cleanly; extracting from recovered data", file_path
            )

        errors: list[str] = []
        features: dict[str, float] = {}
        features |= self._safe_extract(
            "perplexity", _PERPLEXITY_KEYS, lambda: self.perplexity(source_code), errors
        )
        features |= self._safe_extract(
            "stylometric", _STYLOMETRIC_KEYS, lambda: self.stylometric(parsed), errors
        )
        features |= self._safe_extract(
            "distribution",
            _DISTRIBUTION_KEYS,
            lambda: self.distribution(parsed),
            errors,
        )

        return FileAnalysisResult(
            file_path=file_path,
            feature_vector=FeatureVector(**features),
            ai_confidence=0.0,  # placeholder; set by the classifier
            lines_of_code=parsed.lines_of_code,
            is_parseable=parsed.is_valid,
            error_message="; ".join(errors) if errors else None,
        )

    def extract_batch(
        self, files: list[tuple[str, Path]]
    ) -> list[FileAnalysisResult]:
        """Extract features for several files, one at a time.

        Deliberately sequential: parallelism belongs at the commit level, where
        the work is coarse enough to amortise process startup and the language
        model is not being contended for.

        Args:
            files: ``(source_code, file_path)`` pairs.

        Returns:
            One :class:`FileAnalysisResult` per input, in the same order. A file
            that fails outright still gets a zeroed row carrying its error, so
            the output length always matches the input length.
        """
        total = len(files)
        results: list[FileAnalysisResult] = []
        for index, (source_code, file_path) in enumerate(files, start=1):
            logger.info("Processing file %d/%d: %s", index, total, file_path)
            try:
                results.append(self.extract_file_features(source_code, file_path))
            except Exception as exc:  # noqa: BLE001 - one file must not sink the batch
                logger.error("Unexpected failure processing %s: %s", file_path, exc)
                results.append(self._zeroed_result(file_path, f"unhandled: {exc}"))
        return results

    def extract_features_only(
        self, source_code: str, file_path: Path
    ) -> FeatureVector | None:
        """Return just the feature vector, or ``None`` if the file is unusable.

        Used when building the training set, where a partially-zeroed vector is
        worse than no sample at all: it would teach the classifier that a parse
        failure looks like a particular authorship signature. So anything less
        than a clean parse with all three families succeeding is discarded.

        Args:
            source_code: Raw Python source of one file.
            file_path: Path the source came from (reporting only).

        Returns:
            The 16-D :class:`FeatureVector`, or ``None`` when the file did not
            parse cleanly or any feature family failed.
        """
        try:
            result = self.extract_file_features(source_code, file_path)
        except Exception as exc:  # noqa: BLE001 - callers expect None, not a raise
            logger.warning("Feature extraction failed for %s: %s", file_path, exc)
            return None

        if not result.is_parseable or result.error_message is not None:
            logger.warning(
                "Discarding %s: %s",
                file_path,
                result.error_message or "source did not parse",
            )
            return None
        return result.feature_vector

    @staticmethod
    def _safe_extract(
        family: str,
        keys: tuple[str, ...],
        extract: Callable[[], dict[str, float]],
        errors: list[str],
    ) -> dict[str, float]:
        """Run one feature family, substituting zeros for all of it on failure.

        Args:
            family: Family name, used in the log line and the error message.
            keys: The feature names this family is responsible for.
            extract: Thunk invoking the family's extractor.
            errors: Accumulator; a description is appended on failure.

        Returns:
            A dict over exactly ``keys`` — the extracted values, or all zeros.
        """
        try:
            values = extract()
            # A short dict would surface far downstream as a pydantic error on
            # FeatureVector, so an incomplete family is treated as a failed one.
            missing = [key for key in keys if key not in values]
            if missing:
                raise FeatureExtractionError(
                    f"{family} extractor omitted {', '.join(missing)}"
                )
            return {key: float(values[key]) for key in keys}
        except Exception as exc:  # noqa: BLE001 - isolate this family, keep the rest
            logger.warning("%s feature extraction failed: %s", family, exc)
            errors.append(f"{family}: {exc}")
            return dict.fromkeys(keys, 0.0)

    @staticmethod
    def _zeroed_result(file_path: Path, error_message: str) -> FileAnalysisResult:
        """Build an all-zero, unparseable result for a file that yielded nothing."""
        return FileAnalysisResult(
            file_path=file_path,
            feature_vector=FeatureVector(**dict.fromkeys(_ALL_FEATURE_NAMES, 0.0)),
            ai_confidence=0.0,
            lines_of_code=0,
            is_parseable=False,
            error_message=error_message,
        )

    def __call__(self, source_code: str, file_path: Path) -> FileAnalysisResult:
        """Convenience: source code in, :class:`FileAnalysisResult` out."""
        return self.extract_file_features(source_code, file_path)


__all__ = ["FeatureExtractionPipeline"]

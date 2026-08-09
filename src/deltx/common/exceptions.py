"""Custom exception hierarchy for Deltx.

Every error raised deliberately by Deltx derives from :class:`DeltxError`, so
callers can catch the whole family without resorting to a bare ``except``.
"""


class DeltxError(Exception):
    """Base class for every Deltx-specific error."""


class ConfigurationError(DeltxError):
    """Raised when configuration values are missing, malformed, or inconsistent."""


class CheckpointError(DeltxError):
    """Raised when a model checkpoint cannot be fetched, read, or loaded.

    Covers download failures, missing files, and — most importantly — state-dict
    shape or key mismatches, which indicate the published checkpoint no longer
    matches the architecture Deltx builds for it.
    """


class ModelNotLoadedError(DeltxError):
    """Raised when inference is attempted before the detector is loaded."""


class DetectionError(DeltxError):
    """Raised when a single source file cannot be scored."""


class AggregationError(DeltxError):
    """Raised when file-level results cannot be aggregated to a commit score."""


class ExtractionError(DeltxError):
    """Raised when Stage 1 cannot extract commit history from a repository."""


class GitError(ExtractionError):
    """Raised when a Git command fails or a repository cannot be traversed.

    Carries the failed command and its stderr so the caller can report exactly
    which invocation broke rather than a generic non-zero exit.
    """

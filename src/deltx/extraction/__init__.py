"""Stage 1: repository history extraction.

Clones a Python repository and encodes every commit on its primary branch as a
row of the extraction table, carrying the commit's LOC-weighted
``ai_confidence_pct`` from the Stage 2 detector. Chronological, gap-free, and
resumable — the shape the downstream time-series stage requires.
"""

from deltx.extraction.git_history import (
    CommitMeta,
    GitRepository,
    clone_repository,
)
from deltx.extraction.models import CommitRow
from deltx.extraction.pipeline import CommitAiConfidenceExtractor

__all__ = [
    "CommitAiConfidenceExtractor",
    "CommitMeta",
    "CommitRow",
    "GitRepository",
    "clone_repository",
]

"""Feature extraction families for AI detection.

Re-exports the three extractors so callers can reach them from one place:

* :class:`PerplexityExtractor` — F1–F6, surprisal against a code language model
* :class:`StylometricExtractor` — F7–F12, code style and AST shape
* :class:`DistributionExtractor` — F13–F16, token frequency statistics
"""

from deltx.detection.features.distribution import DistributionExtractor
from deltx.detection.features.perplexity import PerplexityExtractor
from deltx.detection.features.stylometric import StylometricExtractor

__all__ = [
    "DistributionExtractor",
    "PerplexityExtractor",
    "StylometricExtractor",
]

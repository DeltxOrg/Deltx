"""AI authorship detection module.

Public surface of Stage 2. The lightweight data models are exported eagerly, but
the three heavyweight classes — :class:`AIDetectionInference`,
:class:`DetectionClassifier` and :class:`FeatureExtractionPipeline` — are loaded
lazily on first attribute access (PEP 562 ``__getattr__``).

The laziness is deliberate, not incidental: importing any of those three pulls in
the torch / transformers / xgboost stack, and ``deltx.detection.dataset`` is
designed to be importable (and its unification path testable) *without* paying
that cost. Eagerly importing them here would drag the torch chain into every
``import deltx.detection.dataset``. Consumers that actually want inference get the
full cost only when they reach for it::

    from deltx.detection import AIDetectionInference   # imports torch, on demand
    from deltx.detection import FileAnalysisResult     # cheap, always available
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from deltx.detection.models import (
    CommitAnalysisResult,
    FeatureVector,
    FileAnalysisResult,
)

if TYPE_CHECKING:
    # Resolved by type checkers and IDEs without triggering the runtime import.
    from deltx.detection.classifier import DetectionClassifier
    from deltx.detection.inference import AIDetectionInference
    from deltx.detection.pipeline import FeatureExtractionPipeline

# Attribute name → module that defines it. Kept out of the module namespace until
# requested, so importing this package stays free of the torch import chain.
_LAZY_EXPORTS: dict[str, str] = {
    "AIDetectionInference": "deltx.detection.inference",
    "DetectionClassifier": "deltx.detection.classifier",
    "FeatureExtractionPipeline": "deltx.detection.pipeline",
}

__all__ = [
    "AIDetectionInference",
    "CommitAnalysisResult",
    "DetectionClassifier",
    "FeatureExtractionPipeline",
    "FeatureVector",
    "FileAnalysisResult",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module __getattr__ returns Any by protocol
    """Import and return a lazily-exported class on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Include the lazy exports in ``dir()`` for discoverability."""
    return sorted(__all__)

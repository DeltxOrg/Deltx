"""Stylometric feature family (F7–F12) for AI authorship detection.

Derives six code-style features directly from a :class:`ParsedSource` — no
language model is involved, so this extractor is fast and fully offline. The
features capture surface-level authorship signals (naming habits, indentation
uniformity, commenting, structural complexity) that tend to differ between
human- and LLM-authored Python:

======  ==========================  ============================================
ID      Name                        Definition
======  ==========================  ============================================
F7      Avg Identifier Length       mean char length of identifiers
F8      Identifier Diversity        unique identifiers / total identifiers
F9      Whitespace Consistency      sample std-dev of indentation widths
F10     Comment-to-Code Ratio       comment lines / lines of code
F11     AST Depth (Mean)            mean nesting depth of AST nodes
F12     AST Node-Type Diversity     Shannon entropy of node-type frequencies
======  ==========================  ============================================

Every feature is defined to degrade gracefully: an invalid parse, an empty
source, or a distribution too small to support the statistic all yield ``0.0``
rather than raising, so the resulting vector is always well-formed.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

from deltx.detection.models import ParsedSource

logger = logging.getLogger(__name__)

# Ordered F7–F12 keys; also the all-zero result returned for degenerate input.
_FEATURE_KEYS: tuple[str, ...] = (
    "f7_avg_identifier_length",
    "f8_identifier_diversity",
    "f9_whitespace_consistency",
    "f10_comment_to_code_ratio",
    "f11_ast_depth_mean",
    "f12_ast_node_type_diversity",
)


def _is_uninformative_identifier(name: str) -> bool:
    """Return True for identifiers excluded from the F7 length statistic.

    Single-underscore throwaways (``_``) and dunder names (``__init__``,
    ``__name__``, …) are language/convention vocabulary rather than
    author-chosen names, so their lengths would bias the mean.
    """
    if name == "_":
        return True
    return name.startswith("__") and name.endswith("__")


class StylometricExtractor:
    """Extracts code style features F7–F12 from parsed Python source."""

    def extract_features(self, parsed: ParsedSource) -> dict[str, float]:
        """Compute all stylometric features.

        Args:
            parsed: A :class:`ParsedSource` from
                :class:`~deltx.detection.parser.PythonSourceParser`.

        Returns:
            A dict keyed by ``f7_avg_identifier_length`` …
            ``f12_ast_node_type_diversity``. An invalid parse or a source with
            no lines of code yields all zeros.
        """
        # An unparseable or code-free file carries no reliable style signal.
        if not parsed.is_valid or parsed.lines_of_code == 0:
            return dict.fromkeys(_FEATURE_KEYS, 0.0)

        return {
            "f7_avg_identifier_length": self._avg_identifier_length(
                parsed.identifiers
            ),
            "f8_identifier_diversity": self._identifier_diversity(
                parsed.identifiers
            ),
            "f9_whitespace_consistency": self._whitespace_consistency(
                parsed.indent_levels
            ),
            "f10_comment_to_code_ratio": (
                parsed.comment_lines / parsed.lines_of_code
            ),
            "f11_ast_depth_mean": self._ast_depth_mean(parsed.ast_depths),
            "f12_ast_node_type_diversity": self._node_type_entropy(
                parsed.ast_node_types
            ),
        }

    @staticmethod
    def _avg_identifier_length(identifiers: list[str]) -> float:
        """F7: mean character length of meaningful identifiers.

        Single-underscore and dunder names are excluded (see
        :func:`_is_uninformative_identifier`). Returns ``0.0`` when no
        qualifying identifiers remain.
        """
        lengths = [
            len(name)
            for name in identifiers
            if not _is_uninformative_identifier(name)
        ]
        if not lengths:
            return 0.0
        return float(np.mean(lengths))

    @staticmethod
    def _identifier_diversity(identifiers: list[str]) -> float:
        """F8: ratio of unique identifiers to total identifier occurrences.

        Lower values indicate heavier reuse of previously defined names, which
        is characteristic of human code that builds on its own abstractions.
        Returns ``0.0`` when there are no identifiers.
        """
        if not identifiers:
            return 0.0
        return len(set(identifiers)) / len(identifiers)

    @staticmethod
    def _whitespace_consistency(indent_levels: list[int]) -> float:
        """F9: sample standard deviation (ddof=1) of indentation widths.

        Lower values indicate more uniform indentation. Requires at least two
        lines for the sample statistic to be defined; otherwise ``0.0``.
        """
        if len(indent_levels) < 2:
            return 0.0
        return float(np.std(indent_levels, ddof=1))

    @staticmethod
    def _ast_depth_mean(ast_depths: list[int]) -> float:
        """F11: mean nesting depth across AST nodes; ``0.0`` when empty."""
        if not ast_depths:
            return 0.0
        return float(np.mean(ast_depths))

    @staticmethod
    def _node_type_entropy(node_types: list[str]) -> float:
        """F12: Shannon entropy (bits) of the AST node-type distribution.

        ``H = −∑ pᵢ log₂ pᵢ`` over node-type frequencies. A single node type
        (or none) has no diversity, so it returns ``0.0``.
        """
        if len(set(node_types)) < 2:
            return 0.0
        total = len(node_types)
        counts = Counter(node_types)
        return -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )

    def __call__(self, parsed: ParsedSource) -> dict[str, float]:
        """Convenience: parsed source in, F7–F12 feature dict out."""
        return self.extract_features(parsed)

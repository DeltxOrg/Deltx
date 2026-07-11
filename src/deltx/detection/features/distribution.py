"""Statistical distribution feature family (F13–F16) for AI authorship detection.

Treats a source file as a bag (and a chain) of lexical tokens and measures the
shape of the resulting frequency distribution. No language model and no AST are
involved — only ``ParsedSource.tokens`` — so this extractor is fast, offline,
and unaffected by syntax errors that defeat AST construction.

======  ==========================  ============================================
ID      Name                        Definition
======  ==========================  ============================================
F13     Shannon Entropy             H = −∑ p(t) log₂ p(t) over token frequencies
F14     Zipf Coefficient Deviation  |α − 1| for log-freq ~ −α·log-rank
F15     Bigram Repetition Rate      repeated bigram types / total bigram types
F16     Hapax Legomena Ratio        tokens seen exactly once / unique tokens
======  ==========================  ============================================

The discriminative intuition: LLM-generated code draws from a narrower
vocabulary and reuses templated constructs, which lowers entropy (F13), flattens
the frequency-rank curve away from the α ≈ 1 that natural code exhibits (F14),
raises bigram repetition (F15), and suppresses the one-off names that inflate a
human author's hapax ratio (F16).

Every feature degrades gracefully: an empty token list, or a distribution too
small to support the statistic, yields ``0.0`` rather than raising, so the
resulting vector is always well-formed.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
from scipy import stats

from deltx.detection.models import ParsedSource

logger = logging.getLogger(__name__)

# Ordered F13–F16 keys; also the all-zero result returned for degenerate input.
_FEATURE_KEYS: tuple[str, ...] = (
    "f13_shannon_entropy",
    "f14_zipf_coefficient_deviation",
    "f15_bigram_repetition_rate",
    "f16_hapax_legomena_ratio",
)

# Zipf's law predicts frequency ∝ 1/rank, i.e. an exponent of 1. Natural source
# code empirically tracks this (Zhang, 2008), so α is measured as a deviation
# from it rather than reported raw.
_EXPECTED_ZIPF_EXPONENT = 1.0

# A least-squares fit through 2 points is exact and its slope carries no
# distributional information, so F14 needs at least 3 distinct ranks.
_MIN_UNIQUE_TOKENS_FOR_ZIPF = 3


class DistributionExtractor:
    """Extracts statistical distribution features F13–F16 from parsed source."""

    def extract_features(self, parsed: ParsedSource) -> dict[str, float]:
        """Compute all distribution features.

        Args:
            parsed: A :class:`ParsedSource` from
                :class:`~deltx.detection.parser.PythonSourceParser`. Only the
                ``tokens`` field is read.

        Returns:
            A dict keyed by ``f13_shannon_entropy`` …
            ``f16_hapax_legomena_ratio``. A source with no tokens yields all
            zeros.
        """
        tokens = parsed.tokens
        # No tokens means no distribution to describe.
        if not tokens:
            return dict.fromkeys(_FEATURE_KEYS, 0.0)

        # Every feature but F15 is a function of the frequency table alone, so
        # it is built once and shared.
        counts = Counter(tokens)

        return {
            "f13_shannon_entropy": self._shannon_entropy(counts),
            "f14_zipf_coefficient_deviation": self._zipf_deviation(counts),
            "f15_bigram_repetition_rate": self._bigram_repetition_rate(tokens),
            "f16_hapax_legomena_ratio": self._hapax_ratio(counts),
        }

    @staticmethod
    def _shannon_entropy(counts: Counter[str]) -> float:
        """F13: Shannon entropy (bits) of the token frequency distribution.

        ``H = −∑ pᵢ log₂ pᵢ`` where ``pᵢ`` is a token's relative frequency. A
        file drawing on a single token has no uncertainty and returns ``0.0``,
        which falls out of the formula without a special case.
        """
        total = sum(counts.values())
        return -sum(
            (count / total) * math.log2(count / total) for count in counts.values()
        )

    @staticmethod
    def _zipf_deviation(counts: Counter[str]) -> float:
        """F14: absolute deviation of the fitted Zipf exponent from 1.0.

        Ranks the token types by descending frequency and regresses
        ``log(freq)`` on ``log(rank)``. Zipf's law makes the slope ``−α``; the
        feature is ``|α − 1|``.

        Returns ``0.0`` for fewer than three unique tokens, where the fit is
        degenerate. Note that a perfectly flat distribution (all tokens equally
        frequent) fits ``α = 0`` and so scores a deviation of ``1.0`` — the
        maximum flatness signal, not a degenerate case.
        """
        if len(counts) < _MIN_UNIQUE_TOKENS_FOR_ZIPF:
            return 0.0

        frequencies = sorted(counts.values(), reverse=True)
        ranks = np.arange(1, len(frequencies) + 1, dtype=np.float64)

        # The slope is invariant to the log base as long as both axes share it,
        # so the natural log is used for both.
        log_ranks = np.log(ranks)
        log_freqs = np.log(np.asarray(frequencies, dtype=np.float64))

        slope = float(stats.linregress(log_ranks, log_freqs).slope)
        alpha = -slope
        return abs(alpha - _EXPECTED_ZIPF_EXPONENT)

    @staticmethod
    def _bigram_repetition_rate(tokens: list[str]) -> float:
        """F15: fraction of consecutive-bigram *types* occurring more than once.

        Bigrams are the overlapping adjacent token pairs. The rate is measured
        over distinct bigram types, not occurrences: in ``a b a b c`` the three
        types are ``(a,b)``, ``(b,a)``, ``(b,c)``, one of which repeats, giving
        ``1/3``.

        Returns ``0.0`` for fewer than two tokens, where no bigram exists.
        """
        if len(tokens) < 2:
            return 0.0

        # The offset slice is deliberately one shorter, so the zip must not be
        # strict: it stops at the final adjacent pair.
        bigram_counts = Counter(zip(tokens, tokens[1:], strict=False))
        repeated = sum(1 for count in bigram_counts.values() if count > 1)
        return repeated / len(bigram_counts)

    @staticmethod
    def _hapax_ratio(counts: Counter[str]) -> float:
        """F16: fraction of token *types* that appear exactly once.

        The denominator is the number of unique tokens, so the feature reads as
        "what share of this file's vocabulary was used only once" — high for
        human code full of one-off names, lower for templated LLM output.
        """
        hapax = sum(1 for count in counts.values() if count == 1)
        return hapax / len(counts)

    def __call__(self, parsed: ParsedSource) -> dict[str, float]:
        """Convenience: parsed source in, F13–F16 feature dict out."""
        return self.extract_features(parsed)

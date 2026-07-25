"""Perplexity & surprisal feature family (F1–F6) for AI authorship detection.

Computes per-token *surprisal* — ``S(tᵢ) = −log₂ P(tᵢ | t₁…tᵢ₋₁)`` — for a
source file using an autoregressive, Python-specialised code language model
(``Salesforce/codegen-350M-mono``), then reduces the surprisal trace to six
scalar features:

======  =======================  ===============================================
ID      Name                     Definition
======  =======================  ===============================================
F1      Mean Token Surprisal     mean(S)
F2      Surprisal Variance       sample variance of S (ddof=1)
F3      Sequence Perplexity      2 ** mean(S)  (model uncertainty; ≥ 1)
F4      Max Surprisal            max(S)
F5      Low-Surprisal Ratio      fraction of S below ``low_surprisal_threshold``
F6      Surprisal Slope          OLS slope of S over token position
======  =======================  ===============================================

The language model is loaded lazily on first inference so that the pure feature
maths (:meth:`PerplexityExtractor.extract_features`) can be exercised — and
unit-tested — without a multi-hundred-megabyte model download.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ModelNotLoadedError
from deltx.detection.models import SurprisalTrace

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Ordered F1–F6 keys; also the all-zero result returned for degenerate traces.
_FEATURE_KEYS: tuple[str, ...] = (
    "f1_mean_surprisal",
    "f2_surprisal_variance",
    "f3_sequence_perplexity",
    "f4_max_surprisal",
    "f5_low_surprisal_ratio",
    "f6_surprisal_slope",
)

# A surprisal trace needs at least this many tokens to yield ≥ 2 surprisal
# values (the first token has no left context and is never scored).
_MIN_TOKENS = 3

# Natural-log base for converting model log-probabilities (nats) into bits.
_LN2 = math.log(2)


class PerplexityExtractor:
    """Extracts surprisal features F1–F6 via a pre-trained autoregressive code LM."""

    def __init__(self, config: DeltxConfig) -> None:
        """Initialise the extractor.

        The heavyweight language model is *not* loaded here; it is fetched on the
        first call to :meth:`compute_surprisal_trace`. This keeps construction and
        the pure feature maths cheap and offline.

        Args:
            config: Global configuration supplying the model name, cache
                directory, device selection, and the low-surprisal threshold.
        """
        self.config = config
        self.device = self._resolve_device(config.device)
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Resolve ``config.device`` to a concrete torch device string."""
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    # -- model loading ------------------------------------------------------

    def _load_model(self) -> None:
        """Download (or load from cache) the tokenizer and model onto the device.

        Uses half precision on CUDA to reduce memory; full precision on CPU
        (float16 matmul is largely unsupported there). The model is put in eval
        mode; gradients are disabled per-call via :func:`torch.no_grad`.
        """
        # Imported lazily so the module stays importable — and the feature maths
        # testable — without the (heavy) transformers dependency being resolved.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading language model %s onto %s", self.config.model_name, self.device
        )
        cache_dir = str(self.config.model_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, cache_dir=cache_dir
        )
        if self.device == "cuda":
            model = model.half()
        model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _ensure_loaded(self) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Return the loaded ``(tokenizer, model)`` pair, loading them if needed."""
        if self._model is None or self._tokenizer is None:
            self._load_model()
        if self._model is None or self._tokenizer is None:  # pragma: no cover
            raise ModelNotLoadedError(
                f"Failed to load language model {self.config.model_name}"
            )
        return self._tokenizer, self._model

    # -- surprisal computation ---------------------------------------------

    def compute_surprisal_trace(self, source_code: str) -> SurprisalTrace:
        """Compute per-token surprisal (in bits) for a source-code string.

        The source is tokenised **without truncation**, so every token is scored.
        A file that fits the context window (``config.max_sequence_length``, itself
        clamped to the model's own maximum context) runs in a single forward pass;
        a longer file is scored with a strided sliding window
        (:meth:`_sliding_window_surprisal`) rather than being cut at the window.
        Surprisal for token *i* is ``−log₂ P(tᵢ | t₁…tᵢ₋₁)``; the first token
        carries no left context and is excluded, so a sequence of ``n`` tokens
        yields ``n − 1`` values regardless of how many windows it took.

        Args:
            source_code: Raw Python source of a single file.

        Returns:
            A :class:`SurprisalTrace`. For inputs shorter than
            :data:`_MIN_TOKENS` tokens the ``surprisal_values`` list is empty.
        """
        tokenizer, model = self._ensure_loaded()

        # CodeGen uses a GPT-2-style BPE tokenizer that does not prepend a BOS
        # token, so every entry here is a genuine source token. No truncation: the
        # full sequence is kept and, if it exceeds the window, scored in windows.
        encoding = tokenizer(source_code, return_tensors="pt", truncation=False)
        input_ids = encoding["input_ids"].to(self.device)
        token_count = int(input_ids.shape[1])

        if token_count < _MIN_TOKENS:
            return SurprisalTrace(
                surprisal_values=[],
                token_count=token_count,
                model_name=self.config.model_name,
            )

        window = self._resolve_window(model)
        if token_count <= window:
            surprisal_values = self._surprisal_from_ids(input_ids)
        else:
            surprisal_values = self._sliding_window_surprisal(input_ids, window)
        return SurprisalTrace(
            surprisal_values=surprisal_values,
            token_count=token_count,
            model_name=self.config.model_name,
        )

    def _resolve_window(self, model: PreTrainedModel) -> int:
        """The context window per forward pass: config value, clamped to the model.

        Feeding more tokens than the model's positional limit raises an indexing
        error, so ``config.max_sequence_length`` is capped at the model's own
        maximum context (``n_positions`` / ``max_position_embeddings``) when the
        model exposes one. Test stubs without a ``config`` fall through to the
        configured value unchanged.
        """
        configured = int(self.config.max_sequence_length)
        model_config = getattr(model, "config", None)
        if model_config is None:
            return configured
        model_max = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )
        if model_max is not None and configured > int(model_max):
            logger.warning(
                "max_sequence_length %d exceeds the model's %d-token context; "
                "clamping the window to %d",
                configured,
                int(model_max),
                int(model_max),
            )
            return int(model_max)
        return configured

    def _sliding_window_surprisal(
        self, input_ids: torch.Tensor, window: int
    ) -> list[float]:
        """Score a sequence longer than the window with a strided sliding window.

        Each step feeds a ``window``-sized slice to the model but keeps surprisal
        only for the tokens *not* already scored by a previous window, so every
        token is scored exactly once and (past the first window) with at least
        ``window − stride`` tokens of real left context. The per-window scores are
        stitched into one contiguous trace covering absolute positions ``1…n−1``.

        Args:
            input_ids: Token ids, shape ``(1, n)`` with ``n > window``.
            window: The per-forward-pass context size (already model-clamped).

        Returns:
            The full per-token surprisal list, length ``n − 1``.
        """
        # Keep at least one token of overlap so each window shares context with the
        # next; a stride ≥ window would leave boundary tokens context-starved.
        stride = max(1, min(int(self.config.surprisal_stride), window - 1))
        seq_len = int(input_ids.shape[1])

        surprisals: list[float] = []
        prev_end = 0  # absolute index (exclusive) already covered by earlier windows
        for begin in range(0, seq_len, stride):
            end = min(begin + window, seq_len)
            window_surprisal = self._surprisal_from_ids(input_ids[:, begin:end])
            # window_surprisal[j] scores absolute target position begin + j + 1.
            # Emit only targets at absolute positions ≥ prev_end (the new ones).
            first_new_abs = max(prev_end, begin + 1)
            start_idx = first_new_abs - begin - 1
            surprisals.extend(window_surprisal[start_idx:])
            prev_end = end
            if end == seq_len:
                break
        return surprisals

    def _surprisal_from_ids(self, input_ids: torch.Tensor) -> list[float]:
        """Run the forward pass, falling back to CPU on CUDA out-of-memory."""
        try:
            return self._forward_surprisal(input_ids)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA out of memory; falling back to CPU for this file (%d tokens)",
                int(input_ids.shape[1]),
            )
            torch.cuda.empty_cache()
            _, model = self._ensure_loaded()
            self._model = model.float().to("cpu")
            self.device = "cpu"
            return self._forward_surprisal(input_ids.to("cpu"))

    def _forward_surprisal(self, input_ids: torch.Tensor) -> list[float]:
        """Single forward pass → list of per-token surprisal values in bits."""
        _, model = self._ensure_loaded()
        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq_len, vocab]

        # log-softmax in float32 for numerical stability even if logits are fp16.
        log_probs = torch.log_softmax(logits.float(), dim=-1)  # nats

        # The distribution at position i-1 predicts the token at position i.
        targets = input_ids[0, 1:]  # [seq_len - 1]
        predicted = log_probs[0, :-1, :]  # [seq_len - 1, vocab]
        token_log_probs = predicted.gather(1, targets.unsqueeze(1)).squeeze(1)  # nats

        # Convert nats → bits: S(tᵢ) = −log₂ P = −ln P / ln 2.
        surprisal_bits = -token_log_probs / _LN2
        return surprisal_bits.tolist()

    # -- feature reduction --------------------------------------------------

    def extract_features(self, surprisal_trace: SurprisalTrace) -> dict[str, float]:
        """Reduce a surprisal trace to the six F1–F6 scalar features.

        Args:
            surprisal_trace: Output of :meth:`compute_surprisal_trace`.

        Returns:
            A dict keyed by ``f1_mean_surprisal`` … ``f6_surprisal_slope``. Traces
            with fewer than two surprisal values (statistics undefined) yield all
            zeros.
        """
        values = np.asarray(surprisal_trace.surprisal_values, dtype=np.float64)
        if values.size < 2:
            return dict.fromkeys(_FEATURE_KEYS, 0.0)

        mean = float(np.mean(values))
        positions = np.arange(values.size, dtype=np.float64)
        below_threshold = values < self.config.low_surprisal_threshold

        return {
            "f1_mean_surprisal": mean,
            "f2_surprisal_variance": float(np.var(values, ddof=1)),
            # Perplexity = 2 ** (mean surprisal in bits); ≥ 1 for non-negative S.
            "f3_sequence_perplexity": float(np.exp2(mean)),
            "f4_max_surprisal": float(np.max(values)),
            "f5_low_surprisal_ratio": float(np.mean(below_threshold)),
            "f6_surprisal_slope": float(np.polyfit(positions, values, deg=1)[0]),
        }

    def __call__(self, source_code: str) -> dict[str, float]:
        """Convenience: source code in, F1–F6 feature dict out."""
        trace = self.compute_surprisal_trace(source_code)
        return self.extract_features(trace)

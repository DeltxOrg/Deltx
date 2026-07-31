"""Source code → authorship probability distribution.

Wraps the loaded DroidDetect checkpoint with the tokenization and chunking
policy the model's measured behaviour demands.
"""

import ast
import logging
from dataclasses import dataclass

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DetectionError
from deltx.detection.modeling import TLModel, load_detector
from deltx.detection.models import ClassDistribution

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredSource:
    """The result of scoring one source string.

    Attributes:
        distribution: Token-count-weighted class distribution over all chunks.
        token_count: Total tokens scored, special tokens excluded.
        chunk_count: Number of forward passes the source required.
    """

    distribution: ClassDistribution
    token_count: int
    chunk_count: int


def _top_level_segments(source: str) -> list[str] | None:
    """Split source into complete top-level blocks, preserving every line.

    Segments are cut at the start of each top-level statement (decorators
    included), so every segment begins at column 0. Leading comments and
    shebangs are absorbed into the first segment, and trailing content into the
    last, so concatenating the segments reproduces the input exactly.

    Args:
        source: Python source text.

    Returns:
        The segments, or ``None`` if the source does not parse or is empty.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    if not tree.body:
        return None

    lines = source.splitlines(keepends=True)
    starts: list[int] = []
    for node in tree.body:
        start = node.lineno
        for decorator in getattr(node, "decorator_list", []):
            start = min(start, decorator.lineno)
        starts.append(start)
    # Absorb any shebang, licence header, or module comment above the first
    # statement rather than dropping it.
    starts[0] = 1

    segments: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else len(lines)
        segment = "".join(lines[start - 1 : end])
        if segment:
            segments.append(segment)
    return segments or None


def _token_windows(token_ids: list[int], budget: int, stride: int) -> list[list[int]]:
    """Slide a fixed window over a token sequence that cannot be split cleanly.

    Args:
        token_ids: Tokens to window, special tokens excluded.
        budget: Maximum tokens per window.
        stride: New tokens consumed per step.

    Returns:
        One or more windows covering the whole sequence.
    """
    if len(token_ids) <= budget:
        return [token_ids]
    windows: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        windows.append(token_ids[start : start + budget])
        if start + budget >= len(token_ids):
            break
        start += stride
    return windows


def split_into_chunks(
    source: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    stride: int,
) -> list[list[int]]:
    """Split source into scoreable token chunks, each within the context window.

    Chunk boundaries fall on top-level statements wherever possible. This is a
    correctness requirement, not a nicety: the model treats indented, mid-scope
    fragments as out-of-distribution and scores them as machine-generated with
    high confidence, so a naive token window would manufacture false positives
    on every chunk after the first.

    Falls back to a sliding token window only when the source does not parse, or
    when a single top-level block is itself larger than the window.

    Args:
        source: Python source text.
        tokenizer: The detector's tokenizer.
        max_tokens: Context window, special tokens included.
        stride: New tokens per step in the fallback window.

    Returns:
        Token ID sequences, each ready to be wrapped in special tokens.
    """
    special = tokenizer.num_special_tokens_to_add()
    budget = max(max_tokens - special, 1)
    stride = max(min(stride, budget), 1)

    whole = tokenizer(source, add_special_tokens=False)["input_ids"]
    if not whole:
        return []
    if len(whole) <= budget:
        return [whole]

    segments = _top_level_segments(source)
    if segments is None:
        logger.debug("unparseable or empty source; falling back to token windows")
        return _token_windows(whole, budget, stride)

    chunks: list[list[int]] = []
    current: list[int] = []
    for segment in segments:
        ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        if len(ids) > budget:
            # A single top-level block overflows the window. Flush what we have,
            # then window this block; its later windows are unavoidably
            # mid-scope, which is why chunk_count is surfaced on the result.
            if current:
                chunks.append(current)
                current = []
            chunks.extend(_token_windows(ids, budget, stride))
            continue
        if current and len(current) + len(ids) > budget:
            chunks.append(current)
            current = []
        current.extend(ids)
    if current:
        chunks.append(current)
    return chunks or _token_windows(whole, budget, stride)


class DroidDetector:
    """Scores Python source with DroidDetect-Base."""

    def __init__(
        self,
        model: TLModel,
        tokenizer: PreTrainedTokenizerBase,
        config: DeltxConfig,
    ) -> None:
        """Wrap an already-loaded model and tokenizer.

        Args:
            model: The loaded detector, in eval mode.
            tokenizer: Its tokenizer.
            config: Active configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_config(cls, config: DeltxConfig | None = None) -> "DroidDetector":
        """Download and load the detector, then wrap it.

        Args:
            config: Active configuration. A default is built when omitted.

        Returns:
            A ready detector.
        """
        config = config or DeltxConfig()
        model, tokenizer = load_detector(config)
        return cls(model, tokenizer, config)

    def _forward(self, token_ids: list[int]) -> list[float]:
        """Score one chunk, returning class probabilities.

        No padding is applied: the model pools with an unmasked mean, so a
        padded batch would make each prediction depend on its neighbours.
        """
        wrapped = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        device = self.config.resolved_device
        input_ids = torch.tensor([wrapped], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities: list[float] = torch.softmax(logits, dim=-1)[0].tolist()
        return probabilities

    def score_source(self, source: str) -> ScoredSource:
        """Score one source string end to end.

        Chunks are combined by token-count weighting, matching how file scores
        are later combined into a commit score.

        Args:
            source: Python source text.

        Returns:
            The weighted distribution and the token and chunk counts behind it.

        Raises:
            DetectionError: If the source yields no scoreable tokens, or too few
                to score reliably.
        """
        chunks = split_into_chunks(
            source,
            self.tokenizer,
            self.config.max_sequence_length,
            self.config.chunk_stride,
        )
        if not chunks:
            msg = "source produced no tokens to score"
            raise DetectionError(msg)

        token_total = sum(len(chunk) for chunk in chunks)
        if token_total < self.config.min_tokens_to_score:
            msg = (
                f"only {token_total} tokens; below the "
                f"{self.config.min_tokens_to_score}-token floor for a reliable score"
            )
            raise DetectionError(msg)

        totals = [0.0, 0.0, 0.0, 0.0]
        total_tokens = 0
        for chunk in chunks:
            weight = len(chunk)
            for index, probability in enumerate(self._forward(chunk)):
                totals[index] += probability * weight
            total_tokens += weight

        averaged = [value / total_tokens for value in totals]
        # Renormalise away float drift so the distribution validator passes.
        scale = sum(averaged)
        averaged = [value / scale for value in averaged]

        return ScoredSource(
            distribution=ClassDistribution.from_probabilities(averaged),
            token_count=total_tokens,
            chunk_count=len(chunks),
        )

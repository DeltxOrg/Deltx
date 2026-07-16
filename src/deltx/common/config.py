"""Runtime configuration for Deltx.

Every field is overridable via a ``DELTX_``-prefixed environment variable or a
``.env`` file, e.g. ``DELTX_DEVICE=cuda`` or ``DELTX_BATCH_SIZE=16``.
"""

from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from deltx.common.constants import (
    DROIDDETECT_REPO,
    MAX_CONTEXT_TOKENS,
    MODERNBERT_REPO,
)

Device = Literal["auto", "cpu", "cuda"]


class DeltxConfig(BaseSettings):
    """Configuration for the Deltx pipeline.

    Attributes:
        detector_repo: HuggingFace repo ID of the DroidDetect checkpoint.
        encoder_repo: HuggingFace repo ID of the upstream encoder whose config
            shapes the ``text_encoder`` container before weights are loaded.
        model_cache_dir: Local cache directory for downloaded model artifacts.
        device: Torch device; ``auto`` picks CUDA when available, else CPU.
        max_sequence_length: Tokens per forward pass, capped at the encoder's
            native context window.
        chunk_stride: New tokens consumed per chunk when a single top-level
            block overflows the window. Overlap is ``max_sequence_length -
            chunk_stride``.
        min_tokens_to_score: Files below this token count are left unscored
            rather than scored unreliably. Below roughly ten tokens the output
            is erratic, not merely uncertain: a docstring-only ``__init__.py``
            measures 0.406 while a single import line measures 0.981, with
            nothing to distinguish them in kind. 10 tokens matches the
            minimum-length filter this project already applied to corpora.
        random_seed: Seeds any sampling performed by the pipeline.

    Note:
        There is deliberately no batch-size setting. The model pools with an
        unmasked mean, so padding a batch changes every member's prediction;
        scoring is one sequence per forward pass to keep results independent of
        batch composition.
    """

    model_config = SettingsConfigDict(
        env_prefix="DELTX_",
        env_file=".env",
        extra="forbid",
        # `model_` is a Pydantic-protected namespace; `model_cache_dir` is ours.
        protected_namespaces=(),
    )

    detector_repo: str = DROIDDETECT_REPO
    encoder_repo: str = MODERNBERT_REPO
    model_cache_dir: Path = Path("data/models/droiddetect")
    device: Device = "auto"
    max_sequence_length: int = Field(default=MAX_CONTEXT_TOKENS, gt=0)
    chunk_stride: int = Field(default=MAX_CONTEXT_TOKENS // 2, gt=0)
    min_tokens_to_score: int = Field(default=10, ge=0)
    random_seed: int = 42

    @field_validator("max_sequence_length")
    @classmethod
    def _cap_sequence_length(cls, value: int) -> int:
        """Clamp the window to the encoder's native context rather than failing.

        A longer request cannot be honoured by the positional embeddings, and
        silently truncating mid-file would be worse than capping here.
        """
        return min(value, MAX_CONTEXT_TOKENS)

    @model_validator(mode="after")
    def _check_stride(self) -> "DeltxConfig":
        """Reject a stride wider than the window, which would skip tokens."""
        if self.chunk_stride > self.max_sequence_length:
            msg = (
                f"chunk_stride ({self.chunk_stride}) exceeds max_sequence_length "
                f"({self.max_sequence_length}); chunks would skip tokens entirely"
            )
            raise ValueError(msg)
        return self

    @property
    def resolved_device(self) -> torch.device:
        """The concrete torch device to run on, resolving ``auto``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def chunk_overlap(self) -> int:
        """Tokens of context each chunk shares with the previous one."""
        return self.max_sequence_length - self.chunk_stride

class ScoringConfig(BaseSettings):
    """Configuration for the Squale quality scoring module."""

    model_config = {
        "env_prefix": "DELTX_SCORING_",
        "env_file": ".env",
        "extra": "ignore",
    }

    sonar_base_url: str = "http://localhost:9000"
    sonar_token: str = ""
    sonar_component_key: str = ""
    normalizer_path: Path = Path("data/scoring/normalizer.json")
    hyperparams_path: Path = Path("data/scoring/hyperparams.json")
    churn_lookback_commits: int = 50
    pagerank_alpha: float = 0.85
    squale_lambda: float = 30.0


"""Tests for configuration and the exception hierarchy."""

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from deltx.common.config import DeltxConfig
from deltx.common.constants import MAX_CONTEXT_TOKENS
from deltx.common.exceptions import (
    AggregationError,
    CheckpointError,
    ConfigurationError,
    DeltxError,
    DetectionError,
    ModelNotLoadedError,
)


class TestExceptions:
    @pytest.mark.parametrize(
        "exc_type",
        [
            ConfigurationError,
            CheckpointError,
            ModelNotLoadedError,
            DetectionError,
            AggregationError,
        ],
    )
    def test_every_error_derives_from_base(self, exc_type: type[Exception]) -> None:
        """One `except DeltxError` must catch the whole family."""
        with pytest.raises(DeltxError):
            raise exc_type("boom")


class TestDeltxConfig:
    def test_defaults(self) -> None:
        config = DeltxConfig()
        assert config.detector_repo == "project-droid/DroidDetect-Base-Binary"
        assert config.max_sequence_length == MAX_CONTEXT_TOKENS
        assert config.random_seed == 42

    def test_sequence_length_is_capped_not_rejected(self) -> None:
        """A too-large window clamps: positional embeddings cannot honour more."""
        config = DeltxConfig(max_sequence_length=999_999)
        assert config.max_sequence_length == MAX_CONTEXT_TOKENS

    def test_stride_wider_than_window_is_rejected(self) -> None:
        """A stride past the window would skip tokens outright."""
        with pytest.raises(ValidationError, match="chunk_stride"):
            DeltxConfig(max_sequence_length=512, chunk_stride=1024)

    def test_chunk_overlap_is_window_minus_stride(self) -> None:
        config = DeltxConfig(max_sequence_length=1000, chunk_stride=600)
        assert config.chunk_overlap == 400

    def test_explicit_device_wins(self) -> None:
        assert DeltxConfig(device="cpu").resolved_device == torch.device("cpu")

    def test_auto_device_resolves_to_a_real_device(self) -> None:
        resolved = DeltxConfig(device="auto").resolved_device
        assert resolved.type in {"cpu", "cuda"}

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every field is settable via a DELTX_-prefixed variable."""
        monkeypatch.setenv("DELTX_RANDOM_SEED", "7")
        monkeypatch.setenv("DELTX_DEVICE", "cpu")
        config = DeltxConfig()
        assert config.random_seed == 7
        assert config.device == "cpu"

    def test_unknown_field_is_rejected(self) -> None:
        """`extra="forbid"` turns a typo'd setting into an error, not a silent no-op."""
        with pytest.raises(ValidationError):
            DeltxConfig(not_a_real_setting=1)

    def test_cache_dir_is_a_path(self, tmp_path: Path) -> None:
        config = DeltxConfig(model_cache_dir=tmp_path)
        assert isinstance(config.model_cache_dir, Path)

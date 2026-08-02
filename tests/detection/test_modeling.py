"""Tests for the architecture definition and the checkpoint loading contract."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deltx.common.config import DeltxConfig
from deltx.common.constants import (
    EXPECTED_CHECKPOINT_TENSORS,
    EXPECTED_MODEL_TENSORS,
    NUM_CLASSES,
    PROJECTION_DIM,
    TEXT_EMBEDDING_DIM,
    TRAINING_ONLY_PREFIX,
)
from deltx.common.exceptions import CheckpointError
from deltx.detection import modeling
from deltx.detection.modeling import TLModel


class StubEncoder(nn.Module):
    """A ModernBERT stand-in producing correctly shaped hidden states."""

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        batch, seq = input_ids.shape
        return SimpleNamespace(
            last_hidden_state=torch.ones(batch, seq, TEXT_EMBEDDING_DIM)
        )


class TestTLModel:
    def test_head_matches_the_shipped_weights_not_the_config(self) -> None:
        """config.json claims projection_dim 128; the weights say 256.

        Building at 128 fails to load, so this guards the value that matters.
        """
        model = TLModel(StubEncoder())
        assert model.text_projection.weight.shape == (
            PROJECTION_DIM,
            TEXT_EMBEDDING_DIM,
        )
        assert model.classifier.weight.shape == (NUM_CLASSES, PROJECTION_DIM)
        assert PROJECTION_DIM == 256

    def test_forward_returns_one_logit_per_class(self) -> None:
        model = TLModel(StubEncoder())
        logits = model(
            input_ids=torch.ones(2, 16, dtype=torch.long),
            attention_mask=torch.ones(2, 16, dtype=torch.long),
        )
        assert logits.shape == (2, NUM_CLASSES)

    def test_pooling_is_an_unmasked_mean(self) -> None:
        """The published model averages every position, padding included.

        A mask-aware mean would be a different model with a different operating
        point, so this behaviour is pinned deliberately.
        """
        model = TLModel(StubEncoder())
        masked = model(
            input_ids=torch.ones(1, 8, dtype=torch.long),
            attention_mask=torch.zeros(1, 8, dtype=torch.long),
        )
        unmasked = model(
            input_ids=torch.ones(1, 8, dtype=torch.long),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
        )
        assert torch.allclose(masked, unmasked)

    def test_expected_tensor_counts_are_documented(self) -> None:
        """272 in the file; 138 once the triplet-loss alias is filtered out."""
        assert EXPECTED_CHECKPOINT_TENSORS == 272
        assert EXPECTED_MODEL_TENSORS == 138
        assert EXPECTED_CHECKPOINT_TENSORS - EXPECTED_MODEL_TENSORS == 134


class TestLoadDetector:
    @staticmethod
    def _head_weights() -> dict[str, torch.Tensor]:
        """The four head tensors a strict load into ``StubEncoder`` requires."""
        return {
            "text_projection.weight": torch.zeros(PROJECTION_DIM, TEXT_EMBEDDING_DIM),
            "text_projection.bias": torch.zeros(PROJECTION_DIM),
            "classifier.weight": torch.zeros(NUM_CLASSES, PROJECTION_DIM),
            "classifier.bias": torch.zeros(NUM_CLASSES),
        }

    def _patch_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        state: dict[str, torch.Tensor],
    ) -> DeltxConfig:
        """Point ``load_detector`` at a local state dict and a stub encoder."""
        weights = tmp_path / "pytorch_model.bin"
        torch.save(state, weights)
        monkeypatch.setattr(modeling, "hf_hub_download", lambda *a, **k: str(weights))
        monkeypatch.setattr(
            modeling.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
        )
        monkeypatch.setattr(modeling, "_build_encoder", lambda config: StubEncoder())
        return DeltxConfig(model_cache_dir=tmp_path, device="cpu")

    def test_training_only_keys_are_filtered_before_a_strict_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The checkpoint ships the triplet-loss module; a strict load must
        still succeed, because those keys are dropped rather than tolerated."""
        state = self._head_weights()
        state[f"{TRAINING_ONLY_PREFIX}sentence_embedder.layers.0.attn.Wo.weight"] = (
            torch.zeros(TEXT_EMBEDDING_DIM, TEXT_EMBEDDING_DIM)
        )
        state[f"{TRAINING_ONLY_PREFIX}sentence_embedder.final_norm.weight"] = (
            torch.zeros(TEXT_EMBEDDING_DIM)
        )

        config = self._patch_download(monkeypatch, tmp_path, state)
        model, _ = modeling.load_detector(config)

        assert not any(
            key.startswith(TRAINING_ONLY_PREFIX) for key in model.state_dict()
        )
        assert model.classifier.weight.shape == (NUM_CLASSES, PROJECTION_DIM)

    def test_a_missing_head_tensor_still_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Filtering must not weaken the guarantee it is filtering around: an
        absent classifier bias would otherwise stay randomly initialised."""
        state = self._head_weights()
        del state["classifier.bias"]

        config = self._patch_download(monkeypatch, tmp_path, state)
        with pytest.raises(CheckpointError, match="does not match the architecture"):
            modeling.load_detector(config)

    def test_download_failure_becomes_checkpoint_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(*args: object, **kwargs: object) -> str:
            raise OSError("network down")

        monkeypatch.setattr(modeling, "hf_hub_download", boom)
        config = DeltxConfig(model_cache_dir=tmp_path, device="cpu")
        with pytest.raises(CheckpointError, match="could not fetch"):
            modeling.load_detector(config)

    def test_shape_mismatch_becomes_checkpoint_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A revised checkpoint must fail loudly, not load partially."""
        weights = tmp_path / "pytorch_model.bin"
        torch.save({"classifier.weight": torch.zeros(4, 999)}, weights)

        monkeypatch.setattr(modeling, "hf_hub_download", lambda *a, **k: str(weights))
        monkeypatch.setattr(
            modeling.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
        )
        monkeypatch.setattr(modeling, "_build_encoder", lambda config: StubEncoder())

        config = DeltxConfig(model_cache_dir=tmp_path, device="cpu")
        with pytest.raises(CheckpointError, match="does not match the architecture"):
            modeling.load_detector(config)

    def test_encoder_build_failure_becomes_checkpoint_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(*args: object, **kwargs: object) -> object:
            raise OSError("no such repo")

        monkeypatch.setattr(modeling.AutoConfig, "from_pretrained", boom)
        config = DeltxConfig(model_cache_dir=tmp_path, device="cpu")
        with pytest.raises(CheckpointError, match="could not build encoder"):
            modeling._build_encoder(config)

"""DroidDetect-Base-Binary architecture definition and checkpoint loading.

The published checkpoint cannot be loaded by ``AutoModelForSequenceClassification``:
its ``config.json`` declares ``model_type: "custom_model"`` and
``architectures: ["Model"]``, ships no ``auto_map``, and the repo contains no
modeling module. The architecture is therefore reconstructed here, mirroring the
model card's ``TLModel`` minus its training-time loss, and the weights are loaded
into it directly.

Neither published value describing the head is correct: ``config.json`` and the
model card both claim a projection width of 128, while the shipped weights are
768 -> 256 -> 2. The export also carries the training-time triplet loss, whose
tensors are filtered out at load time. Both facts were established by reading
``pytorch_model.bin``, not its documentation.
"""

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F  # noqa: N812
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from deltx.common.config import DeltxConfig
from deltx.common.constants import (
    CHECKPOINT_FILENAME,
    EXPECTED_CHECKPOINT_TENSORS,
    NUM_CLASSES,
    PROJECTION_DIM,
    TEXT_EMBEDDING_DIM,
    TRAINING_ONLY_PREFIX,
)
from deltx.common.exceptions import CheckpointError

logger = logging.getLogger(__name__)


class TLModel(nn.Module):
    """The DroidDetect classification head over a ModernBERT encoder.

    Mirrors the published ``TLModel`` for the forward path only. The training
    objective (cross-entropy plus a batch-hard triplet loss) is omitted: it
    contributed no weights to the checkpoint, and reproducing it would pull in
    ``sentence-transformers`` for no inference benefit.

    Attributes:
        text_encoder: The ModernBERT backbone.
        text_projection: ``Linear(768 -> 256)``.
        classifier: ``Linear(256 -> 2)``.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        projection_dim: int = PROJECTION_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Build the head.

        Args:
            text_encoder: An instantiated ModernBERT-compatible encoder.
            projection_dim: Width of the projection layer. Defaults to the
                checkpoint's true width, which is *not* the value its
                ``config.json`` advertises.
            num_classes: Number of authorship classes.
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.text_projection = nn.Linear(TEXT_EMBEDDING_DIM, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Score a batch of tokenized samples.

        Pooling is an *unmasked* mean over every position, padding included.
        That is what the published model does, so it is what determines the
        reported operating point; a mask-aware mean would be a different model.
        The consequence is that padded batches are not equivalent to
        single-sample scoring, which is why the detector avoids padding rather
        than "fixing" the pooling here.

        Args:
            input_ids: Token IDs, shape ``(batch, seq)``.
            attention_mask: Attention mask, shape ``(batch, seq)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = hidden.mean(dim=1)
        projected = F.relu(self.text_projection(pooled))
        logits: torch.Tensor = self.classifier(projected)
        return logits


def _build_encoder(config: DeltxConfig) -> nn.Module:
    """Instantiate an empty ModernBERT container for the checkpoint weights.

    The checkpoint's ``text_encoder.*`` keys match upstream ModernBERT-base
    exactly, so the upstream config alone is enough to shape the container;
    upstream weights are never downloaded.

    Args:
        config: Active Deltx configuration.

    Returns:
        An uninitialised encoder module.

    Raises:
        CheckpointError: If the encoder config cannot be fetched or built.
    """
    try:
        encoder_config = AutoConfig.from_pretrained(
            config.encoder_repo, cache_dir=config.model_cache_dir
        )
        # ModernBERT's `reference_compile` path invokes torch.compile, which is
        # brittle on Windows/CPU and buys nothing for single-sample inference.
        encoder_config.reference_compile = False
        encoder: nn.Module = AutoModel.from_config(encoder_config)
        return encoder
    except (OSError, ValueError) as exc:
        msg = f"could not build encoder from {config.encoder_repo!r}: {exc}"
        raise CheckpointError(msg) from exc


def load_detector(
    config: DeltxConfig | None = None,
) -> tuple[TLModel, PreTrainedTokenizerBase]:
    """Download and load DroidDetect-Base-Binary, ready for inference.

    The checkpoint ships the training-time triplet-loss module alongside the
    model proper: 272 tensors, of which 134 sit under ``additional_loss.*`` and
    alias the encoder's own storages. Those keys are dropped, and the remaining
    138 are loaded with ``strict=True``.

    Filtering one known prefix is preferred over a blanket ``strict=False``.
    Strictness on what remains still guarantees every projection and classifier
    weight came from the checkpoint, rather than silently leaving the head at
    its random initialisation; and it will surface any upstream revision that
    changes shape.

    Args:
        config: Active configuration. A default is constructed when omitted.

    Returns:
        The loaded model in eval mode on the configured device, and its tokenizer.

    Raises:
        CheckpointError: If the checkpoint cannot be fetched, is malformed, or
            does not match the architecture built for it.
    """
    config = config or DeltxConfig()
    config.model_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading detector %s", config.detector_repo)
    try:
        weights_path = Path(
            hf_hub_download(
                config.detector_repo,
                CHECKPOINT_FILENAME,
                cache_dir=config.model_cache_dir,
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.detector_repo, cache_dir=config.model_cache_dir
        )
    except OSError as exc:
        msg = f"could not fetch {config.detector_repo!r}: {exc}"
        raise CheckpointError(msg) from exc

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if len(state_dict) != EXPECTED_CHECKPOINT_TENSORS:
        logger.warning(
            "checkpoint has %d tensors, expected %d — the published artifact "
            "may have been revised",
            len(state_dict),
            EXPECTED_CHECKPOINT_TENSORS,
        )

    weights = {
        key: tensor
        for key, tensor in state_dict.items()
        if not key.startswith(TRAINING_ONLY_PREFIX)
    }
    logger.debug(
        "dropped %d training-time tensors under %r",
        len(state_dict) - len(weights),
        TRAINING_ONLY_PREFIX,
    )

    model = TLModel(_build_encoder(config))
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        msg = (
            f"checkpoint does not match the architecture built for it: {exc}\n"
            "This usually means the published projection width changed; it is "
            "256 in the weights despite config.json claiming 128."
        )
        raise CheckpointError(msg) from exc

    model.eval()
    model.to(config.resolved_device)
    logger.info(
        "Detector ready on %s (%d tensors loaded)",
        config.resolved_device,
        len(weights),
    )
    return model, tokenizer

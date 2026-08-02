"""End-to-end tests against the real checkpoint.

Marked ``slow``: these download ~600 MB on first run and are excluded from the
default suite. Run with ``poetry run pytest -m slow``.

They exist to catch a specific class of silent failure. Nothing in the published
artifact pins the output index order — ``config.json`` carries no ``id2label``,
and the model card is not a reliable substitute, since the same card misstates
the projection width. If a checkpoint revision reordered the classes, every
other test would still pass while ``ai_confidence_pct`` came out inverted.
"""

import pytest
import torch

from deltx.common.config import DeltxConfig
from deltx.common.constants import (
    EXPECTED_MODEL_TENSORS,
    NUM_CLASSES,
    PROJECTION_DIM,
    TEXT_EMBEDDING_DIM,
)
from deltx.detection.detector import DroidDetector
from deltx.detection.models import DroidLabel

pytestmark = pytest.mark.slow

# Written by an LLM, so AI-authored by construction.
AI_SOURCE = '''
def binary_search(arr: list[int], target: int) -> int:
    """
    Perform a binary search on a sorted list to find the target value.

    Args:
        arr (list[int]): A sorted list of integers to search through.
        target (int): The value to search for.

    Returns:
        int: The index of the target if found, otherwise -1.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


@pytest.fixture(scope="module")
def detector() -> DroidDetector:
    """The real detector, cached for the module."""
    return DroidDetector.from_config(DeltxConfig(device="cpu"))


def test_checkpoint_loads_strictly(detector: DroidDetector) -> None:
    """A strict load proves no weight was silently left uninitialised.

    The model holds only the 138 tensors it uses: the file's 134
    ``additional_loss.*`` keys are filtered out before loading.
    """
    state = detector.model.state_dict()
    assert len(state) == EXPECTED_MODEL_TENSORS
    assert not any(key.startswith("additional_loss.") for key in state)
    assert state["text_projection.weight"].shape == (
        PROJECTION_DIM,
        TEXT_EMBEDDING_DIM,
    )
    assert state["classifier.weight"].shape == (NUM_CLASSES, PROJECTION_DIM)


def test_index_zero_is_human(detector: DroidDetector) -> None:
    """Complete pre-LLM stdlib modules must score as human.

    Guards the index order. If the two classes were reordered upstream, this
    file would score near 1.0 instead of near 0.0.
    """
    import sysconfig
    from pathlib import Path

    source = (Path(sysconfig.get_paths()["stdlib"]) / "textwrap.py").read_text(
        encoding="utf-8"
    )
    result = detector.score_source(source)
    assert result.distribution.predicted_label is DroidLabel.HUMAN_GENERATED
    assert result.distribution.p_ai < 0.5


def test_index_one_is_machine(detector: DroidDetector) -> None:
    """The positive direction of the same guard."""
    result = detector.score_source(AI_SOURCE)
    assert result.distribution.predicted_label is DroidLabel.MACHINE_GENERATED
    assert result.distribution.p_ai > 0.5


def test_the_two_directions_are_separated(detector: DroidDetector) -> None:
    """Both guards passing by a hair would not prove the order.

    A model whose head was reordered, or whose classifier never loaded, tends
    to sit near the decision boundary on both inputs. Requiring the human and
    machine samples to land on opposite sides *with* a margin is what makes the
    pair informative rather than coincidental.
    """
    import sysconfig
    from pathlib import Path

    human = detector.score_source(
        (Path(sysconfig.get_paths()["stdlib"]) / "textwrap.py").read_text(
            encoding="utf-8"
        )
    )
    machine = detector.score_source(AI_SOURCE)
    assert machine.distribution.p_ai - human.distribution.p_ai > 0.5


def test_scoring_is_independent_of_batch_composition(
    detector: DroidDetector,
) -> None:
    """Padding contaminates the unmasked mean pool.

    This asserts *why* the detector never pads: scoring the same source inside a
    padded batch gives a different answer, so single-sequence scoring is the
    only composition-independent option.
    """
    ids = detector.tokenizer(AI_SOURCE, add_special_tokens=False)["input_ids"]
    wrapped = detector.tokenizer.build_inputs_with_special_tokens(ids)

    single = torch.tensor([wrapped], dtype=torch.long)
    with torch.no_grad():
        alone = detector.model(input_ids=single, attention_mask=torch.ones_like(single))

    pad_id = detector.tokenizer.pad_token_id or 0
    padded = torch.tensor([wrapped + [pad_id] * 64], dtype=torch.long)
    mask = torch.ones_like(padded)
    mask[:, len(wrapped) :] = 0
    with torch.no_grad():
        in_batch = detector.model(input_ids=padded, attention_mask=mask)

    assert not torch.allclose(alone, in_batch, atol=1e-3)


def test_long_file_chunks_and_stays_in_range(detector: DroidDetector) -> None:
    """A file past the context window must chunk and still yield a valid score."""
    import sysconfig
    from pathlib import Path

    source = (Path(sysconfig.get_paths()["stdlib"]) / "argparse.py").read_text(
        encoding="utf-8"
    )
    result = detector.score_source(source)
    assert result.chunk_count > 1
    assert 0.0 <= result.distribution.p_ai <= 1.0
    assert sum(result.distribution.model_dump().values()) == pytest.approx(1.0)

"""Project-wide constants.

Values here are facts about external artifacts (the published checkpoint, the
tokenizer, the Python language) rather than tunable settings. Anything a user
might reasonably want to override belongs in :mod:`deltx.common.config` instead.
"""

from typing import Final

# --- Model identity -------------------------------------------------------

#: HuggingFace repo holding the DroidDetect-Base detector checkpoint.
DROIDDETECT_REPO: Final = "project-droid/DroidDetect-Base"

#: Upstream encoder the checkpoint's ``text_encoder.*`` weights were trained on.
#: Its key set matches the checkpoint exactly (134 tensors, no divergence), so it
#: can be instantiated from config and loaded into directly.
MODERNBERT_REPO: Final = "answerdotai/ModernBERT-base"

#: Weight file shipped by the detector repo. It ships no safetensors variant.
CHECKPOINT_FILENAME: Final = "pytorch_model.bin"

# --- Architecture ---------------------------------------------------------

#: Hidden size of ModernBERT-base, and the input width of the projection layer.
TEXT_EMBEDDING_DIM: Final = 768

#: Width of the projection layer.
#:
#: The published ``config.json`` claims 128 and so does the model card's
#: ``TLModel.__init__`` default, but the shipped weights are
#: ``text_projection.weight (256, 768)`` and ``classifier.weight (4, 256)``.
#: 256 is what the checkpoint actually contains; building at 128 fails on a
#: shape mismatch. Measured directly from ``pytorch_model.bin``.
PROJECTION_DIM: Final = 256

#: DroidDetect predicts four authorship classes. See ``DroidLabel``.
NUM_CLASSES: Final = 4

#: ModernBERT's native context window, and the tokenizer's ``model_max_length``.
MAX_CONTEXT_TOKENS: Final = 8192

#: Tensor count in a well-formed checkpoint: 134 encoder + 2 projection + 2
#: classifier. A different count means the published artifact changed.
EXPECTED_CHECKPOINT_TENSORS: Final = 138

# --- File selection -------------------------------------------------------

#: Deltx analyses Python only.
PYTHON_SUFFIX: Final = ".py"

#: Filenames skipped as packaging or test boilerplate rather than authored code.
SKIPPED_FILENAMES: Final[frozenset[str]] = frozenset({"setup.py", "conftest.py"})

#: Any path containing one of these directory components is skipped.
SKIPPED_DIR_PARTS: Final[frozenset[str]] = frozenset({"__pycache__"})

# --- Output range ---------------------------------------------------------

#: ``ai_confidence_pct`` bounds. 0 = high confidence human, 100 = high
#: confidence AI. Index [4] of the 15-D commit vector.
AI_CONFIDENCE_MIN: Final = 0.0
AI_CONFIDENCE_MAX: Final = 100.0

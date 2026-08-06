"""Project-wide constants.

Values here are facts about external artifacts (the published checkpoint, the
tokenizer, the Python language) rather than tunable settings. Anything a user
might reasonably want to override belongs in :mod:`deltx.common.config` instead.
"""

from typing import Final

# --- Model identity -------------------------------------------------------

#: HuggingFace repo holding the DroidDetect binary detector checkpoint.
DROIDDETECT_REPO: Final = "project-droid/DroidDetect-Base-Binary"

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
#: ``text_projection.weight (256, 768)`` and ``classifier.weight (2, 256)``.
#: 256 is what the checkpoint actually contains; building at 128 fails on a
#: shape mismatch. Measured directly from ``pytorch_model.bin``.
PROJECTION_DIM: Final = 256

#: DroidDetect-Base-Binary predicts two authorship classes. See ``DroidLabel``.
NUM_CLASSES: Final = 2

#: ModernBERT's native context window, and the tokenizer's ``model_max_length``.
MAX_CONTEXT_TOKENS: Final = 8192

#: Key prefix of the training-time triplet-loss module. Its 134 tensors alias
#: the same storages as ``text_encoder.*`` — the loss simply held a reference to
#: the encoder — so the file is byte-for-byte the size of a single backbone and
#: dropping the prefix discards nothing. Inference never constructs the loss.
TRAINING_ONLY_PREFIX: Final = "additional_loss."

#: Tensor count in a well-formed checkpoint file: 134 encoder + 134 triplet-loss
#: alias + 2 projection + 2 classifier. A different count means the published
#: artifact changed.
EXPECTED_CHECKPOINT_TENSORS: Final = 272

#: Tensors remaining once :data:`TRAINING_ONLY_PREFIX` is filtered out — the set
#: ``TLModel`` actually loads, and the size of its own ``state_dict``.
EXPECTED_MODEL_TENSORS: Final = 138

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

# --- Stage 1: history extraction ------------------------------------------

#: Text encodings tried, in order, when decoding a file blob from a commit.
#: UTF-8 is the norm for Python source; latin-1 is a total decoder that never
#: raises, so it is the last resort before a blob is treated as binary.
BLOB_ENCODINGS: Final[tuple[str, ...]] = ("utf-8", "latin-1")

#: Commits processed between checkpoint writes to the output Parquet. Small
#: enough that an interrupted run over a large repository loses little work,
#: large enough that rewriting the whole frame is not the bottleneck. Resume
#: continues from the last checkpointed commit.
CHECKPOINT_INTERVAL: Final = 25

#: Ordered column schema of the extraction output Parquet. The order is part of
#: the contract downstream stages read against, so it is defined once here.
EXTRACTION_COLUMNS: Final[tuple[str, ...]] = (
    "repo_url",
    "commit_hash",
    "commit_timestamp",
    "commit_author",
    "commit_message",
    "commit_index",
    "files_changed_py",
    "total_loc_scored",
    "ai_confidence_pct",
    "file_scores_json",
)

"""Application configuration via Pydantic BaseSettings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class DeltxConfig(BaseSettings):
    """Global configuration for the Deltx pipeline."""

    model_config = {"env_prefix": "DELTX_", "env_file": ".env", "extra": "ignore"}

    model_name: str = "Salesforce/codegen-350M-mono"
    model_cache_dir: Path = Path("data/models/codegen")
    device: str = "auto"
    low_surprisal_threshold: float = 2.0
    classifier_path: Path = Path("data/models/detector.joblib")
    batch_size: int = 32
    # Per-forward-pass context window for surprisal scoring (clamped to the code
    # LM's own maximum context). Files longer than this are scored with a strided
    # sliding window, not truncated, so every token is processed.
    max_sequence_length: int = 1024
    # New tokens scored per sliding-window step when a file exceeds the window; the
    # window overlap (context each late token sees) is max_sequence_length minus
    # this. Only takes effect for files longer than the window.
    surprisal_stride: int = 512
    confidence_threshold: float = 0.5
    random_seed: int = 42

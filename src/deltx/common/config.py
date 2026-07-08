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
    max_sequence_length: int = 1024
    confidence_threshold: float = 0.5
    random_seed: int = 42

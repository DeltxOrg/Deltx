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


class ScoringConfig(BaseSettings):
    """Configuration for the Squale quality scoring module."""

    model_config = {"env_prefix": "DELTX_SCORING_", "env_file": ".env", "extra": "ignore"}

    sonar_base_url: str = "http://localhost:9000"
    sonar_token: str = ""
    sonar_component_key: str = ""
    normalizer_path: Path = Path("data/scoring/normalizer.json")
    hyperparams_path: Path = Path("data/scoring/hyperparams.json")
    churn_lookback_commits: int = 50
    pagerank_alpha: float = 0.85
    squale_lambda: float = 30.0


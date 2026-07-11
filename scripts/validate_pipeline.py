"""End-to-end validation of the Deltx AI detection pipeline.

Downloads (or reuses a cached copy of) AIGCodeSet, extracts the 16-D feature
vector for 100 samples (50 human, 50 AI) with the real CodeGen language model,
trains the XGBoost classifier with default hyperparameters, evaluates on a 20%
stratified hold-out, prints a rich-formatted report — classification metrics,
SHAP feature ranking, confusion matrix — and saves the trained model to
``data/models/detector.joblib``.

Usage::

    poetry run python scripts/validate_pipeline.py

Notes:
    The first run downloads the AIGCodeSet CSVs (~a few MB) and the
    CodeGen-350M model (~700 MB) if they are not cached under ``data/``.
    Feature extraction for 100 samples takes a few minutes on CPU.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.model_selection import train_test_split

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltx.common.config import DeltxConfig  # noqa: E402
from deltx.common.exceptions import DeltxError  # noqa: E402
from deltx.detection.classifier import DetectionClassifier  # noqa: E402
from deltx.detection.dataset import DatasetManager  # noqa: E402
from deltx.detection.models import FeatureVector  # noqa: E402
from deltx.detection.pipeline import FeatureExtractionPipeline  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

SAMPLES_PER_CLASS = 50
HOLDOUT_FRACTION = 0.2
TOP_FEATURES_SHOWN = 3
FEATURES_CACHE = Path("data/processed/validation_features.parquet")


def load_samples(config: DeltxConfig) -> pd.DataFrame:
    """Load a balanced 50/50 human/AI sample of AIGCodeSet.

    Downloads the dataset on first use; subsequent runs read the cached copy.

    Returns:
        A frame of ``2 * SAMPLES_PER_CLASS`` rows in the unified schema.

    Raises:
        DeltxError: If the dataset cannot be downloaded or yields too few
            samples of either class.
    """
    manager = DatasetManager(config)
    manager.download_aigcodeset()  # no-op when already cached
    unified = manager.load_and_unify(sources=["aigcodeset"])

    parts: list[pd.DataFrame] = []
    for label in (0, 1):
        rows = unified[unified["label"] == label]
        if len(rows) < SAMPLES_PER_CLASS:
            raise DeltxError(
                f"AIGCodeSet yielded only {len(rows)} samples for label={label}; "
                f"need {SAMPLES_PER_CLASS}"
            )
        parts.append(rows.sample(n=SAMPLES_PER_CLASS, random_state=config.random_seed))
    balanced = pd.concat(parts, ignore_index=True)
    console.print(
        f"Sampled [bold]{len(balanced)}[/bold] rows "
        f"({SAMPLES_PER_CLASS} human, {SAMPLES_PER_CLASS} AI) from AIGCodeSet"
    )
    return balanced


def extract_features(config: DeltxConfig, samples: pd.DataFrame) -> pd.DataFrame:
    """Extract the 16 features for every sample, checkpointing to parquet.

    Re-running the script resumes from (or fully reuses) the checkpoint, so the
    expensive language-model pass happens once.
    """
    manager = DatasetManager(config)
    pipeline = FeatureExtractionPipeline(config)
    FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features = manager.extract_features_dataset(
        samples, pipeline, output_path=FEATURES_CACHE
    )
    console.print(
        f"Feature extraction complete: [bold]{len(features)}[/bold] usable rows "
        f"({len(samples) - len(features)} rejected)"
    )
    return features


def train_and_evaluate(
    config: DeltxConfig, features: pd.DataFrame
) -> tuple[DetectionClassifier, dict[str, Any], dict[str, Any]]:
    """Train with default hyperparameters and evaluate on a 20% hold-out."""
    feature_columns = FeatureVector.feature_names()
    matrix = features.loc[:, feature_columns].to_numpy(dtype=float)
    labels = features["label"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
        labels,
        test_size=HOLDOUT_FRACTION,
        stratify=labels,
        random_state=config.random_seed,
    )
    console.print(
        f"Split: [bold]{len(y_train)}[/bold] train / [bold]{len(y_test)}[/bold] test "
        f"(stratified, {HOLDOUT_FRACTION:.0%} hold-out)"
    )

    classifier = DetectionClassifier(config)
    classifier.train(x_train, y_train, tune_hyperparameters=False)
    metrics = classifier.evaluate(x_test, y_test)
    shap_importance = classifier.compute_shap_importance(x_test)
    return classifier, metrics, shap_importance


def render_report(metrics: dict[str, Any], shap_importance: dict[str, Any]) -> None:
    """Print the metrics, SHAP ranking, and confusion matrix as rich tables."""
    metrics_table = Table(title="Classification metrics (20% hold-out)")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    for name in ("accuracy", "f1_score", "auroc", "precision", "recall", "auprc"):
        metrics_table.add_row(name, f"{metrics[name]:.4f}")
    console.print(metrics_table)

    mean_abs = shap_importance["mean_abs_shap"]
    ranking = shap_importance["feature_ranking"]
    shap_table = Table(title="SHAP feature importance (mean |SHAP|)")
    shap_table.add_column("Rank", justify="right")
    shap_table.add_column("Feature")
    shap_table.add_column("Mean |SHAP|", justify="right")
    for rank, name in enumerate(ranking, start=1):
        style = "bold green" if rank <= TOP_FEATURES_SHOWN else ""
        shap_table.add_row(str(rank), name, f"{mean_abs[name]:.4f}", style=style)
    console.print(shap_table)

    top = ", ".join(
        f"{name} ({mean_abs[name]:.4f})" for name in ranking[:TOP_FEATURES_SHOWN]
    )
    console.print(Panel(f"Top {TOP_FEATURES_SHOWN} features: {top}", style="green"))

    cm = metrics["confusion_matrix"]
    cm_table = Table(title="Confusion matrix (rows=true, cols=predicted)")
    cm_table.add_column("")
    cm_table.add_column("pred: human", justify="right")
    cm_table.add_column("pred: ai", justify="right")
    cm_table.add_row("true: human", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("true: ai", str(cm[1][0]), str(cm[1][1]))
    console.print(cm_table)


def main() -> int:
    """Run the full validation workflow. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO)
    config = DeltxConfig()
    console.rule("[bold]Deltx detection pipeline validation")

    try:
        samples = load_samples(config)
        features = extract_features(config, samples)
        classifier, metrics, shap_importance = train_and_evaluate(config, features)
    except DeltxError as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        return 1

    render_report(metrics, shap_importance)
    saved_to = classifier.save()  # defaults to config.classifier_path
    console.print(f"[bold green]Model saved →[/bold green] {saved_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

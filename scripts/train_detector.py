"""Phase C of the production training routine: train, evaluate, and ship.

Reads the extracted feature matrix produced by Phase B
(``data/processed/train_features.parquet``) and:

1. **Headline (in-distribution) evaluation** — a stratified train/val/test split
   with :class:`RandomizedSearchCV` hyperparameter tuning and early stopping,
   reporting hold-out metrics, the 5-fold CV score, a confusion matrix, and SHAP
   feature importances.
2. **Leave-one-model-out (LOMO) evaluation** — a *fresh* model trained on every
   generator except one, then scored on the held-out generator's samples only.
   This is the generalization stress test: can the detector flag an LLM it never
   saw during training?
3. **Ships the production model** — retrains on the full feature set with the
   tuned parameters and saves it to ``data/models/detector.joblib``
   (``config.classifier_path``), the artifact the ``deltx-detect`` CLI loads.

This script trains only on already-extracted numeric features; it never touches
the language model, so it runs comfortably on CPU in minutes.

Usage::

    poetry run python scripts/train_detector.py --holdout-model gemini
    poetry run python scripts/train_detector.py \
        --features data/processed/train_features.parquet \
        --holdout-model codestral --per-class 10000

Pass ``--no-tune`` for a fast dry run with default hyperparameters, and omit
``--holdout-model`` to skip the LOMO stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_FEATURES = Path("data/processed/train_features.parquet")
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
TOP_FEATURES_SHOWN = 5
HUMAN_LABEL = 0
AI_LABEL = 1
FEATURE_COLUMNS = FeatureVector.feature_names()

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Feature parquet from Phase B.",
    )
    parser.add_argument(
        "--holdout-model",
        type=str,
        default=None,
        help="Generator (ai_model value) to hold out for the LOMO test.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="Rebalance to exactly this many rows per class (default: min class).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter search (fast dry run with default params).",
    )
    return parser.parse_args(argv)


def load_features(path: Path) -> pd.DataFrame:
    """Load and validate the extracted feature frame.

    Raises:
        DeltxError: If the file is missing or lacks the expected columns.
    """
    if not path.is_file():
        raise DeltxError(
            f"Feature file not found: {path}. Run Phase B (feature extraction) first."
        )
    frame = pd.read_parquet(path)
    missing = [c for c in (*FEATURE_COLUMNS, "label") if c not in frame.columns]
    if missing:
        raise DeltxError(f"Feature file missing required columns: {missing}")
    return frame


def rebalance(frame: pd.DataFrame, per_class: int | None, seed: int) -> pd.DataFrame:
    """Downsample to an exact, equal per-class count (rejects may have skewed it)."""
    counts = frame["label"].value_counts()
    n_human = int(counts.get(HUMAN_LABEL, 0))
    n_ai = int(counts.get(AI_LABEL, 0))
    if n_human == 0 or n_ai == 0:
        raise DeltxError(f"Need both classes; got human={n_human}, ai={n_ai}")

    target = min(n_human, n_ai) if per_class is None else min(per_class, n_human, n_ai)
    balanced = (
        frame.groupby("label", group_keys=False)
        .sample(n=target, random_state=seed)
        .reset_index(drop=True)
    )
    console.print(
        f"Rebalanced to [bold]{target}[/bold]/class "
        f"(from human={n_human}, ai={n_ai})"
    )
    return balanced


def to_xy(frame: pd.DataFrame) -> tuple[FloatArray, IntArray]:
    """Extract the (features, labels) numpy arrays from a frame."""
    x = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=np.int_)
    return x, y


def run_headline(
    manager: DatasetManager,
    config: DeltxConfig,
    features: pd.DataFrame,
    tune: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Train with tuning + early stopping and evaluate on a stratified hold-out."""
    train_df, test_df = manager.prepare_train_test_split(
        features, test_size=TEST_FRACTION, stratify_by="label"
    )
    train_core, val_df = train_test_split(
        train_df,
        test_size=VAL_FRACTION,
        stratify=train_df["label"],
        random_state=config.random_seed,
        shuffle=True,
    )
    console.print(
        f"Split: [bold]{len(train_core)}[/bold] train / "
        f"[bold]{len(val_df)}[/bold] val / [bold]{len(test_df)}[/bold] test"
    )

    x_train, y_train = to_xy(train_core)
    x_val, y_val = to_xy(val_df)
    x_test, y_test = to_xy(test_df)

    classifier = DetectionClassifier(config)
    train_result = classifier.train(
        x_train, y_train, x_val, y_val, tune_hyperparameters=tune
    )
    metrics = classifier.evaluate(x_test, y_test)
    shap_importance = classifier.compute_shap_importance(x_test)
    return train_result, metrics, shap_importance


def run_lomo(
    manager: DatasetManager,
    config: DeltxConfig,
    features: pd.DataFrame,
    holdout_model: str,
    tune: bool,
) -> dict[str, Any]:
    """Train without one generator, then score on that unseen generator only.

    Returns:
        A dict with the unseen-generator detection recall, the held-out sample
        count, and the full mixed-test metrics for context.
    """
    lomo_train, lomo_test = manager.prepare_train_test_split(
        features, test_size=TEST_FRACTION, holdout_model=holdout_model
    )
    x_train, y_train = to_xy(lomo_train)

    classifier = DetectionClassifier(config)
    classifier.train(x_train, y_train, tune_hyperparameters=tune)

    # Isolate the held-out generator's rows (all AI) for the clean OOD signal.
    models = lomo_test["ai_model"].fillna("").astype(str).str.strip().str.lower()
    gen_mask = models == holdout_model.strip().lower()
    x_gen, _ = to_xy(lomo_test[gen_mask])
    gen_preds = classifier.predict(x_gen)
    unseen_recall = (
        float((gen_preds == AI_LABEL).mean()) if len(x_gen) else float("nan")
    )

    x_test, y_test = to_xy(lomo_test)
    mixed_metrics = classifier.evaluate(x_test, y_test)
    return {
        "holdout_model": holdout_model,
        "unseen_samples": int(gen_mask.sum()),
        "unseen_recall": unseen_recall,
        "mixed_metrics": mixed_metrics,
    }


def ship(
    config: DeltxConfig, features: pd.DataFrame, tune: bool
) -> tuple[Path, dict[str, Any]]:
    """Retrain on the full feature set with tuned params and save the model."""
    x_all, y_all = to_xy(features)
    classifier = DetectionClassifier(config)
    result = classifier.train(x_all, y_all, tune_hyperparameters=tune)
    saved_to = classifier.save()  # defaults to config.classifier_path
    return saved_to, result


def render_report(
    train_result: dict[str, Any],
    metrics: dict[str, Any],
    shap_importance: dict[str, Any],
    lomo: dict[str, Any] | None,
) -> None:
    """Print headline metrics, CV score, confusion matrix, SHAP, and LOMO."""
    metrics_table = Table(title="Headline metrics (stratified 20% hold-out)")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    for name in ("accuracy", "f1_score", "auroc", "precision", "recall", "auprc"):
        metrics_table.add_row(name, f"{metrics[name]:.4f}")
    console.print(metrics_table)

    cv_scores = train_result.get("cv_scores") or {}
    if cv_scores:
        console.print(
            Panel(
                f"CV {cv_scores['scoring']}: {cv_scores['best_score']:.4f} "
                f"(±{cv_scores['best_score_std']:.4f}) over "
                f"{cv_scores['n_splits']} folds, {cv_scores['n_iter']} iters",
                title="Hyperparameter search",
                style="cyan",
            )
        )

    cm = metrics["confusion_matrix"]
    cm_table = Table(title="Confusion matrix (rows=true, cols=predicted)")
    cm_table.add_column("")
    cm_table.add_column("pred: human", justify="right")
    cm_table.add_column("pred: ai", justify="right")
    cm_table.add_row("true: human", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("true: ai", str(cm[1][0]), str(cm[1][1]))
    console.print(cm_table)

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

    if lomo is not None:
        mixed = lomo["mixed_metrics"]
        console.print(
            Panel(
                f"Held-out generator: [bold]{lomo['holdout_model']}[/bold] "
                f"({lomo['unseen_samples']} unseen AI samples)\n"
                f"Unseen-generator detection recall: "
                f"[bold]{lomo['unseen_recall']:.4f}[/bold]\n"
                f"Mixed LOMO test — auroc {mixed['auroc']:.4f}, "
                f"f1 {mixed['f1_score']:.4f}, accuracy {mixed['accuracy']:.4f}",
                title="Leave-one-model-out generalization",
                style="magenta",
            )
        )


def main(argv: list[str] | None = None) -> int:
    """Run the full train/evaluate/ship workflow. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    config = DeltxConfig()
    tune = not args.no_tune
    console.rule("[bold]Phase C — train, evaluate, and ship the detector")

    try:
        manager = DatasetManager(config)
        features = load_features(args.features)
        features = rebalance(features, args.per_class, config.random_seed)

        train_result, metrics, shap_importance = run_headline(
            manager, config, features, tune
        )

        lomo: dict[str, Any] | None = None
        if args.holdout_model is not None:
            lomo = run_lomo(manager, config, features, args.holdout_model, tune)

        saved_to, _ = ship(config, features, tune)
    except DeltxError as exc:
        console.print(f"[red]Training failed:[/red] {exc}")
        return 1

    render_report(train_result, metrics, shap_importance, lomo)
    console.print(f"[bold green]Production model saved →[/bold green] {saved_to}")
    console.print(
        "[dim]Verify with: poetry run deltx-detect analyze --file <some.py>[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

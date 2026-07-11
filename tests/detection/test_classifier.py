"""Tests for the XGBoost detection classifier.

The classifier is exercised on synthetic two-Gaussian data rather than the real
feature pipeline, so the suite stays fast and offline. Hyperparameter tuning is
disabled in most tests (it fits hundreds of trees); the one end-to-end test that
tunes shrinks the search via the module-level knobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ModelNotLoadedError
from deltx.detection import classifier as classifier_module
from deltx.detection.classifier import DetectionClassifier
from deltx.detection.models import FeatureVector

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

FEATURE_NAMES = FeatureVector.feature_names()
_N_PER_CLASS = 100


def _gaussian_blobs(seed: int = 0) -> tuple[FloatArray, IntArray]:
    """Two overlapping 16-D Gaussians: human ~N(0,1), AI ~N(1.5,1).

    The 1.5-sigma per-dimension separation is small enough that the classes
    overlap yet, pooled across 16 dimensions, comfortably separable — so a
    trained model clears the 0.7 accuracy floor without hitting a trivial 1.0.
    """
    rng = np.random.default_rng(seed)
    human = rng.normal(0.0, 1.0, size=(_N_PER_CLASS, 16))
    ai = rng.normal(1.5, 1.0, size=(_N_PER_CLASS, 16))
    features = np.vstack([human, ai])
    labels = np.concatenate(
        [np.zeros(_N_PER_CLASS, dtype=int), np.ones(_N_PER_CLASS, dtype=int)]
    )
    return features, labels


@pytest.fixture
def blobs() -> tuple[FloatArray, IntArray]:
    """A fixed 200-sample synthetic dataset (16 features, two classes)."""
    return _gaussian_blobs()


@pytest.fixture
def split(
    blobs: tuple[FloatArray, IntArray],
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """A stratified 160/40 train/test split of :func:`_gaussian_blobs`."""
    features, labels = blobs
    return train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )


@pytest.fixture
def trained(
    config: DeltxConfig, split: tuple[FloatArray, FloatArray, IntArray, IntArray]
) -> tuple[DetectionClassifier, FloatArray, IntArray]:
    """A classifier trained (without tuning) on the split, plus the test set."""
    X_train, X_test, y_train, y_test = split
    clf = DetectionClassifier(config)
    clf.train(X_train, y_train, tune_hyperparameters=False)
    return clf, X_test, y_test


# -- 1. train and predict ----------------------------------------------------


def test_train_and_predict_beats_accuracy_floor(
    trained: tuple[DetectionClassifier, FloatArray, IntArray],
) -> None:
    """A model trained on separable-but-overlapping data predicts well on test."""
    clf, X_test, y_test = trained
    assert clf.is_fitted

    predictions = clf.predict(X_test)
    accuracy = float((predictions == y_test).mean())
    assert accuracy > 0.7


def test_train_with_validation_set_enables_early_stopping(
    config: DeltxConfig,
    split: tuple[FloatArray, FloatArray, IntArray, IntArray],
) -> None:
    """A validation set wires an early-stopping eval set into the fit."""
    X_train, X_test, y_train, _ = split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    clf = DetectionClassifier(config)
    result = clf.train(X_fit, y_fit, X_val, y_val, tune_hyperparameters=False)

    assert clf.is_fitted
    # Early stopping records the best boosting round on the fitted booster.
    assert clf.model is not None
    assert clf.model.best_iteration is not None
    assert clf.predict(X_test).shape == (len(X_test),)
    assert result["training_time_seconds"] >= 0.0


# -- 2. predict_proba range and shape ----------------------------------------


def test_predict_proba_is_a_probability_per_sample(
    trained: tuple[DetectionClassifier, FloatArray, IntArray],
) -> None:
    clf, X_test, _ = trained
    proba = clf.predict_proba(X_test)

    assert proba.shape == (len(X_test),)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


# -- 3. thresholding ---------------------------------------------------------


def test_predict_applies_confidence_threshold(config: DeltxConfig) -> None:
    """The same probabilities yield different labels at different thresholds."""
    clf = DetectionClassifier(config)
    # Bypass the model: predict() thresholds whatever predict_proba returns.
    fixed = np.array([0.2, 0.4, 0.6, 0.8])
    clf.predict_proba = lambda _X: fixed  # type: ignore[method-assign]
    features = np.zeros((4, 16))

    config.confidence_threshold = 0.3
    lenient = clf.predict(features)
    config.confidence_threshold = 0.7
    strict = clf.predict(features)

    assert lenient.tolist() == [0, 1, 1, 1]
    assert strict.tolist() == [0, 0, 0, 1]
    assert not np.array_equal(lenient, strict)


# -- 4. evaluation -----------------------------------------------------------


def test_evaluate_returns_all_metrics(
    trained: tuple[DetectionClassifier, FloatArray, IntArray],
) -> None:
    clf, X_test, y_test = trained
    metrics = clf.evaluate(X_test, y_test)

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "confusion_matrix",
        "classification_report",
    ):
        assert key in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    # 2x2 confusion matrix over labels [0, 1].
    assert np.array(metrics["confusion_matrix"]).shape == (2, 2)
    assert isinstance(metrics["classification_report"], str)


# -- 5. SHAP importance ------------------------------------------------------


def test_compute_shap_importance_ranks_every_feature(
    trained: tuple[DetectionClassifier, FloatArray, IntArray],
) -> None:
    clf, X_test, _ = trained
    result = clf.compute_shap_importance(X_test)

    assert len(result["feature_ranking"]) == 16
    assert set(result["feature_ranking"]) == set(FEATURE_NAMES)
    assert len(result["mean_abs_shap"]) == 16
    # Ranking is by descending mean absolute SHAP value.
    ranked_values = [
        result["mean_abs_shap"][name] for name in result["feature_ranking"]
    ]
    assert ranked_values == sorted(ranked_values, reverse=True)


def test_compute_shap_importance_subsamples_large_inputs(
    trained: tuple[DetectionClassifier, FloatArray, IntArray],
) -> None:
    """More rows than ``max_samples`` are capped in the returned SHAP array."""
    clf, X_test, _ = trained
    result = clf.compute_shap_importance(X_test, max_samples=10)
    assert result["shap_values"].shape == (10, 16)


# -- 6. save / load roundtrip ------------------------------------------------


def test_save_and_load_roundtrip_preserves_predictions(
    config: DeltxConfig,
    split: tuple[FloatArray, FloatArray, IntArray, IntArray],
    tmp_path: Path,
) -> None:
    X_train, X_test, y_train, _ = split
    original = DetectionClassifier(config)
    original.train(X_train, y_train, tune_hyperparameters=False)
    expected = original.predict_proba(X_test)

    path = tmp_path / "detector.joblib"
    saved = original.save(path)
    assert saved == path
    assert path.exists()

    reloaded = DetectionClassifier(config)
    reloaded.load(path)
    assert reloaded.is_fitted
    assert reloaded.feature_names == FEATURE_NAMES
    np.testing.assert_allclose(reloaded.predict_proba(X_test), expected)


def test_save_defaults_to_configured_path(
    config: DeltxConfig,
    split: tuple[FloatArray, FloatArray, IntArray, IntArray],
    tmp_path: Path,
) -> None:
    """With no argument, save() writes to config.classifier_path."""
    config.classifier_path = tmp_path / "nested" / "detector.joblib"
    X_train, _, y_train, _ = split
    clf = DetectionClassifier(config)
    clf.train(X_train, y_train, tune_hyperparameters=False)

    saved = clf.save()
    assert saved == config.classifier_path
    assert config.classifier_path.exists()


# -- 7. unfitted guards ------------------------------------------------------


def test_predict_before_fit_raises(config: DeltxConfig) -> None:
    clf = DetectionClassifier(config)
    features = np.zeros((3, 16))

    with pytest.raises(ModelNotLoadedError):
        clf.predict(features)
    with pytest.raises(ModelNotLoadedError):
        clf.predict_proba(features)


def test_save_before_fit_raises(config: DeltxConfig, tmp_path: Path) -> None:
    clf = DetectionClassifier(config)
    with pytest.raises(ModelNotLoadedError):
        clf.save(tmp_path / "detector.joblib")


def test_load_missing_file_raises(config: DeltxConfig, tmp_path: Path) -> None:
    clf = DetectionClassifier(config)
    with pytest.raises(ModelNotLoadedError):
        clf.load(tmp_path / "absent.joblib")


# -- 8. end-to-end workflow --------------------------------------------------


def test_train_and_evaluate_end_to_end(
    config: DeltxConfig,
    blobs: tuple[FloatArray, IntArray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_and_evaluate wires train → evaluate → SHAP → save over DataFrames."""
    # Shrink and serialise the search so the tuning path stays quick and avoids
    # spawning worker processes during the test.
    monkeypatch.setattr(classifier_module, "SEARCH_N_ITER", 6)
    monkeypatch.setattr(classifier_module, "SEARCH_N_JOBS", 1)
    config.classifier_path = tmp_path / "detector.joblib"

    features, labels = blobs
    frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    frame["label"] = labels
    train_df, test_df = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame["label"]
    )

    clf, results = DetectionClassifier.train_and_evaluate(
        config,
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

    assert clf.is_fitted
    assert set(results) == {"training", "evaluation", "shap_importance", "model_path"}
    assert results["training"]["best_params"]  # tuning populated the parameters
    assert results["training"]["cv_scores"]["scoring"] == "f1"
    assert 0.0 <= results["evaluation"]["accuracy"] <= 1.0
    assert len(results["shap_importance"]["feature_ranking"]) == 16
    assert Path(results["model_path"]).exists()
    assert results["model_path"] == str(config.classifier_path)

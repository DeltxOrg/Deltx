"""XGBoost classifier for AI authorship detection (Stage 2, step 5).

Consumes the 16-D feature vectors produced by
:class:`~deltx.detection.pipeline.FeatureExtractionPipeline` and learns to
separate human- from LLM-authored Python. The module covers the whole classifier
lifecycle: hyperparameter search, training with optional early stopping,
threshold-based prediction, metric evaluation, SHAP feature attribution, and
joblib persistence.

The probability of AI authorship this classifier emits is the raw material for
``ai_confidence_pct`` — index [4] of the downstream 15-D commit vector — once the
inference layer aggregates it file → commit.

XGBoost 2.x API notes
=====================

The project floors ``xgboost`` at 2.0, and two facts about that line shape the
code below:

* ``eval_metric`` and ``early_stopping_rounds`` are *constructor* arguments, not
  ``fit`` arguments — 2.0 removed them from ``fit``. So early stopping is wired
  in :meth:`DetectionClassifier._build_estimator`, only when a validation set is
  supplied (a plain fit with ``early_stopping_rounds`` set but no ``eval_set``
  would raise).
* ``use_label_encoder`` was removed entirely: XGBoost no longer encodes targets,
  so the flag the design sketch mentioned is neither needed nor accepted here.
  Passing it only provokes an "unused parameter" warning; the labels are already
  ``{0, 1}``.

Hyperparameter search runs with ``refit=False`` — the CV loop finds the best
parameters, then a single final estimator is fit on the full training data (with
early stopping when a validation set is given). That avoids the redundant refit
``RandomizedSearchCV`` would otherwise perform on top of the fit done here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ClassifierError, ModelNotLoadedError
from deltx.detection.models import FeatureVector

logger = logging.getLogger(__name__)

# A rich Table is a renderable, not a string, and cannot round-trip through
# logging's Formatter (which stringifies the record). It is drawn on a Console,
# mirroring how the dataset module renders its progress bars; the same metrics
# are also emitted through `logger` so structured log consumers still see them.
_console = Console()

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

# The binary label contract, shared with the dataset module (0=human, 1=AI).
_HUMAN_LABEL = 0
_AI_LABEL = 1

# -- training configuration ---------------------------------------------------
# Module-level so a test can shrink the search (n_iter, n_jobs) without touching
# the production values.

#: Hyperparameters used when tuning is disabled — a reasonable middle of the
#: search space below.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

#: RandomizedSearchCV sampling space (CLAUDE.md evaluation strategy).
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

SEARCH_N_ITER = 50
SEARCH_CV_FOLDS = 5
SEARCH_SCORING = "f1"
SEARCH_N_JOBS = -1
EARLY_STOPPING_ROUNDS = 20

# Metric keys rendered in the evaluation table, in display order.
_SCALAR_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auroc",
    "auprc",
)


class DetectionClassifier:
    """XGBoost-based classifier for AI code detection."""

    def __init__(self, config: DeltxConfig) -> None:
        """Initialise an untrained classifier.

        Args:
            config: Global configuration; supplies ``random_seed`` (seeds the
                search, folds, and model), ``confidence_threshold`` (the decision
                boundary applied in :meth:`predict`), and ``classifier_path`` (the
                default persistence location).
        """
        self.config = config
        self.model: xgb.XGBClassifier | None = None
        self.is_fitted: bool = False
        self.feature_names: list[str] = FeatureVector.feature_names()

    # -- training ----------------------------------------------------------

    def train(
        self,
        X_train: FloatArray,
        y_train: IntArray,
        X_val: FloatArray | None = None,
        y_val: IntArray | None = None,
        tune_hyperparameters: bool = True,
    ) -> dict[str, Any]:
        """Train the XGBoost classifier.

        When ``tune_hyperparameters`` is set, a :class:`RandomizedSearchCV` over
        :data:`SEARCH_SPACE` (``n_iter`` = :data:`SEARCH_N_ITER`, stratified
        :data:`SEARCH_CV_FOLDS`-fold, scoring :data:`SEARCH_SCORING`) selects the
        parameters; otherwise :data:`DEFAULT_PARAMS` is used. Either way a single
        final estimator is then fit on the full training data.

        When both ``X_val`` and ``y_val`` are given they become an early-stopping
        eval set (patience :data:`EARLY_STOPPING_ROUNDS`); the validation set is
        *not* fed into the CV search, which validates on its own folds.

        Args:
            X_train: Training features, shape ``(n_samples, 16)``.
            y_train: Training labels in ``{0, 1}``, shape ``(n_samples,)``.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels for early stopping.
            tune_hyperparameters: Whether to run the randomized search.

        Returns:
            A dict with ``best_params`` (the parameters used), ``cv_scores`` (the
            search's best score and spread, or ``{}`` when tuning is skipped), and
            ``training_time_seconds``.
        """
        features = np.asarray(X_train, dtype=np.float64)
        labels = np.asarray(y_train).astype(int)
        has_validation = X_val is not None and y_val is not None

        start = time.perf_counter()
        if tune_hyperparameters:
            logger.info(
                "Tuning hyperparameters: RandomizedSearchCV "
                "(n_iter=%d, %d-fold, scoring=%r)",
                SEARCH_N_ITER,
                SEARCH_CV_FOLDS,
                SEARCH_SCORING,
            )
            best_params, cv_scores = self._search_hyperparameters(features, labels)
        else:
            logger.info("Training with default hyperparameters (tuning disabled)")
            best_params = dict(DEFAULT_PARAMS)
            cv_scores = {}

        self.model = self._build_estimator(
            best_params, use_early_stopping=has_validation
        )
        if has_validation:
            eval_features = np.asarray(X_val, dtype=np.float64)
            eval_labels = np.asarray(y_val).astype(int)
            self.model.fit(
                features,
                labels,
                eval_set=[(eval_features, eval_labels)],
                verbose=False,
            )
        else:
            self.model.fit(features, labels)
        self.is_fitted = True
        elapsed = time.perf_counter() - start

        self._log_training_summary(best_params, elapsed, has_validation)
        return {
            "best_params": best_params,
            "cv_scores": cv_scores,
            "training_time_seconds": elapsed,
        }

    def _search_hyperparameters(
        self, X: FloatArray, y: IntArray
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the randomized CV search and return ``(best_params, cv_scores)``.

        Uses ``refit=False``: for a single scorer the best-parameter attributes
        are still populated, and skipping the refit avoids training a model the
        caller immediately replaces with an early-stopped one.
        """
        base = xgb.XGBClassifier(
            random_state=self.config.random_seed,
            eval_metric="logloss",
            enable_categorical=False,
        )
        folds = StratifiedKFold(
            n_splits=SEARCH_CV_FOLDS,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        search = RandomizedSearchCV(
            base,
            SEARCH_SPACE,
            n_iter=SEARCH_N_ITER,
            scoring=SEARCH_SCORING,
            cv=folds,
            random_state=self.config.random_seed,
            n_jobs=SEARCH_N_JOBS,
            refit=False,
            error_score="raise",
        )
        search.fit(X, y)

        best_index = int(search.best_index_)
        cv_scores: dict[str, Any] = {
            "best_score": float(search.best_score_),
            "best_score_std": float(
                search.cv_results_["std_test_score"][best_index]
            ),
            "scoring": SEARCH_SCORING,
            "n_splits": SEARCH_CV_FOLDS,
            "n_iter": SEARCH_N_ITER,
        }
        best_params = dict(search.best_params_)
        logger.info(
            "Best CV %s: %.4f (±%.4f) with %s",
            SEARCH_SCORING,
            cv_scores["best_score"],
            cv_scores["best_score_std"],
            best_params,
        )
        return best_params, cv_scores

    def _build_estimator(
        self, params: dict[str, Any], *, use_early_stopping: bool
    ) -> xgb.XGBClassifier:
        """Construct an XGBClassifier from ``params`` plus the fixed settings.

        ``eval_metric`` and (conditionally) ``early_stopping_rounds`` are set here
        because XGBoost 2.x takes them on the constructor, not on ``fit``.
        """
        kwargs: dict[str, Any] = {
            **params,
            "random_state": self.config.random_seed,
            "eval_metric": "logloss",
            "enable_categorical": False,
        }
        if use_early_stopping:
            kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        return xgb.XGBClassifier(**kwargs)

    def _log_training_summary(
        self, best_params: dict[str, Any], elapsed: float, has_validation: bool
    ) -> None:
        """Emit a one-line summary, noting the early-stopping iteration if any."""
        model = self.model
        best_iteration = getattr(model, "best_iteration", None)
        if has_validation and best_iteration is not None:
            logger.info(
                "Training complete in %.2fs; early-stopped at iteration %d",
                elapsed,
                best_iteration,
            )
        else:
            logger.info("Training complete in %.2fs (params=%s)", elapsed, best_params)

    # -- prediction --------------------------------------------------------

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return the probability of AI authorship in ``[0, 1]`` per sample.

        Args:
            X: Features, shape ``(n_samples, 16)``.

        Returns:
            A 1-D array of positive-class (AI) probabilities, shape
            ``(n_samples,)``.

        Raises:
            ModelNotLoadedError: If the classifier has not been trained or loaded.
        """
        model = self._require_model()
        features = np.asarray(X, dtype=np.float64)
        proba = np.asarray(model.predict_proba(features), dtype=np.float64)
        return proba[:, _AI_LABEL]

    def predict(self, X: FloatArray) -> IntArray:
        """Return binary predictions thresholded at ``config.confidence_threshold``.

        A sample is labelled AI (``1``) when its AI probability is at or above the
        threshold, human (``0``) otherwise. Using the configurable threshold
        rather than XGBoost's fixed 0.5 lets the operating point be tuned without
        retraining.

        Args:
            X: Features, shape ``(n_samples, 16)``.

        Returns:
            Integer predictions in ``{0, 1}``, shape ``(n_samples,)``.

        Raises:
            ModelNotLoadedError: If the classifier has not been trained or loaded.
        """
        proba = self.predict_proba(X)
        return (proba >= self.config.confidence_threshold).astype(int)

    # -- evaluation --------------------------------------------------------

    def evaluate(self, X_test: FloatArray, y_test: IntArray) -> dict[str, Any]:
        """Score the classifier on a labelled test set.

        Thresholded predictions drive the classification metrics; the raw
        probabilities drive the ranking metrics (AUROC/AUPRC). The scalar metrics
        are logged as a rich table.

        Args:
            X_test: Test features, shape ``(n_samples, 16)``.
            y_test: True labels in ``{0, 1}``, shape ``(n_samples,)``.

        Returns:
            A dict with ``accuracy``, ``precision``, ``recall``, ``f1_score``,
            ``auroc``, ``auprc`` (all floats; the ranking metrics are ``nan`` when
            only one class is present), ``confusion_matrix`` (a nested list,
            ordered ``[0, 1]``), and ``classification_report`` (a string).

        Raises:
            ModelNotLoadedError: If the classifier has not been trained or loaded.
        """
        self._require_model()
        y_true = np.asarray(y_test).astype(int)
        proba = self.predict_proba(X_test)
        y_pred = (proba >= self.config.confidence_threshold).astype(int)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, pos_label=_AI_LABEL, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, pos_label=_AI_LABEL, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, pos_label=_AI_LABEL, zero_division=0)
            ),
        }
        # AUROC/AUPRC need both classes present; a single-class test set (a real
        # possibility under leave-one-model-out) makes them undefined.
        if len(np.unique(y_true)) < 2:
            logger.warning(
                "Only one class present in y_test; AUROC and AUPRC are undefined"
            )
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
        else:
            metrics["auroc"] = float(roc_auc_score(y_true, proba))
            metrics["auprc"] = float(average_precision_score(y_true, proba))

        metrics["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=[_HUMAN_LABEL, _AI_LABEL]
        ).tolist()
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=[_HUMAN_LABEL, _AI_LABEL],
            target_names=["human", "ai"],
            zero_division=0,
        )

        self._log_evaluation(metrics)
        return metrics

    @staticmethod
    def _log_evaluation(metrics: dict[str, Any]) -> None:
        """Render the scalar metrics as a rich table and a structured log line."""
        table = Table(title="Detection classifier — evaluation")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        for name in _SCALAR_METRICS:
            table.add_row(name, f"{metrics[name]:.4f}")
        _console.print(table)
        logger.info(
            "Evaluation — acc=%.4f precision=%.4f recall=%.4f f1=%.4f "
            "auroc=%.4f auprc=%.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["auroc"],
            metrics["auprc"],
        )

    # -- explainability ----------------------------------------------------

    def compute_shap_importance(
        self, X: FloatArray, max_samples: int = 1000
    ) -> dict[str, Any]:
        """Compute SHAP feature importance with an exact TreeExplainer.

        ``shap.TreeExplainer`` is exact and fast for tree ensembles and needs no
        background data. For a binary XGBClassifier it returns one SHAP value per
        feature per sample (positive-class log-odds contributions).

        Args:
            X: Features to explain, shape ``(n_samples, 16)``.
            max_samples: Upper bound on rows fed to the explainer; larger inputs
                are randomly subsampled (seeded by ``config.random_seed``) to keep
                the computation bounded.

        Returns:
            A dict with ``mean_abs_shap`` (feature name → mean absolute SHAP
            value), ``shap_values`` (the raw per-sample array, for downstream
            visualisation), and ``feature_ranking`` (feature names sorted by
            importance, descending).

        Raises:
            ModelNotLoadedError: If the classifier has not been trained or loaded.
        """
        model = self._require_model()
        features = np.asarray(X, dtype=np.float64)

        total = features.shape[0]
        if total > max_samples:
            rng = np.random.default_rng(self.config.random_seed)
            selected = rng.choice(total, size=max_samples, replace=False)
            features = features[selected]
            logger.info("SHAP: subsampled %d → %d rows", total, max_samples)

        explainer = shap.TreeExplainer(model)
        shap_values = self._positive_class_shap(explainer.shap_values(features))

        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_abs_shap = {
            name: float(value)
            for name, value in zip(self.feature_names, mean_abs, strict=True)
        }
        feature_ranking = sorted(
            mean_abs_shap, key=lambda name: mean_abs_shap[name], reverse=True
        )
        logger.info(
            "SHAP most important feature: %s (mean|SHAP|=%.4f)",
            feature_ranking[0],
            mean_abs_shap[feature_ranking[0]],
        )
        return {
            "mean_abs_shap": mean_abs_shap,
            "shap_values": shap_values,
            "feature_ranking": feature_ranking,
        }

    @staticmethod
    def _positive_class_shap(raw: FloatArray | list[FloatArray]) -> FloatArray:
        """Reduce a TreeExplainer result to a 2-D positive-class SHAP array.

        Binary XGBoost yields a single ``(n_samples, n_features)`` array, but some
        shap/model combinations return a per-class list or a
        ``(n_samples, n_features, n_classes)`` block; both are collapsed onto the
        positive (AI) class here.
        """
        if isinstance(raw, list):
            chosen = raw[_AI_LABEL] if len(raw) == 2 else raw[-1]
            return np.asarray(chosen, dtype=np.float64)
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim == 3:
            return values[:, :, -1]
        return values

    # -- persistence -------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Persist the trained model (and its feature names) with joblib.

        Args:
            path: Destination file; defaults to ``config.classifier_path``.

        Returns:
            The path written to.

        Raises:
            ModelNotLoadedError: If there is no fitted model to save.
        """
        self._require_model()
        destination = path if path is not None else self.config.classifier_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names}, destination
        )
        logger.info("Saved classifier → %s", destination)
        return destination

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved model, marking the classifier fitted.

        Args:
            path: Source file; defaults to ``config.classifier_path``.

        Raises:
            ModelNotLoadedError: If the file is absent or does not hold a model.
        """
        source = path if path is not None else self.config.classifier_path
        if not source.exists():
            raise ModelNotLoadedError(f"No classifier file at {source}")

        payload = joblib.load(source)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ModelNotLoadedError(f"Malformed classifier file at {source}")

        self.model = payload["model"]
        self.feature_names = payload.get("feature_names", FeatureVector.feature_names())
        self.is_fitted = True
        logger.info("Loaded classifier ← %s", source)

    def _require_model(self) -> xgb.XGBClassifier:
        """Return the fitted model or raise if the classifier is not ready."""
        if self.model is None or not self.is_fitted:
            raise ModelNotLoadedError(
                "Classifier is not fitted; call train() or load() first"
            )
        return self.model

    # -- end-to-end workflow ----------------------------------------------

    @classmethod
    def train_and_evaluate(
        cls,
        config: DeltxConfig,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        label_column: str = "label",
    ) -> tuple[DetectionClassifier, dict[str, Any]]:
        """Run the full train → evaluate → explain → save workflow.

        Args:
            config: Global configuration.
            train_df: Training rows carrying the feature and label columns.
            test_df: Test rows carrying the feature and label columns.
            feature_columns: Feature column names; defaults to the 16 canonical
                :meth:`FeatureVector.feature_names`.
            label_column: Name of the label column.

        Returns:
            The fitted :class:`DetectionClassifier` and a results dict with
            ``training`` (the :meth:`train` return), ``evaluation`` (the
            :meth:`evaluate` return), ``shap_importance`` (the
            :meth:`compute_shap_importance` return, computed on the test set), and
            ``model_path`` (where the model was saved).

        Raises:
            ClassifierError: If either frame is empty or is missing a required
                column.
        """
        columns = (
            feature_columns
            if feature_columns is not None
            else FeatureVector.feature_names()
        )
        X_train, y_train = cls._split_xy(train_df, columns, label_column)
        X_test, y_test = cls._split_xy(test_df, columns, label_column)

        classifier = cls(config)
        training = classifier.train(X_train, y_train, tune_hyperparameters=True)
        evaluation = classifier.evaluate(X_test, y_test)
        shap_importance = classifier.compute_shap_importance(X_test)
        model_path = classifier.save()

        results: dict[str, Any] = {
            "training": training,
            "evaluation": evaluation,
            "shap_importance": shap_importance,
            "model_path": str(model_path),
        }
        return classifier, results

    @staticmethod
    def _split_xy(
        df: pd.DataFrame, feature_columns: list[str], label_column: str
    ) -> tuple[FloatArray, IntArray]:
        """Split a frame into an ``(X, y)`` pair, validating the schema first.

        Raises:
            ClassifierError: If ``df`` is empty or is missing any feature column
                or the label column.
        """
        missing = [name for name in feature_columns if name not in df.columns]
        if missing:
            raise ClassifierError(
                f"DataFrame is missing feature column(s): {', '.join(missing)}"
            )
        if label_column not in df.columns:
            raise ClassifierError(
                f"DataFrame is missing the label column {label_column!r}"
            )
        if df.empty:
            raise ClassifierError("Cannot train or evaluate on an empty DataFrame")

        matrix = df.loc[:, feature_columns].to_numpy(dtype=float)
        X = np.asarray(matrix, dtype=np.float64)
        y = np.asarray(df[label_column].to_numpy(), dtype=int)
        return X, y


__all__ = ["DetectionClassifier"]

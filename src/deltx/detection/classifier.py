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
  ``fit`` arguments — 2.0 removed them from ``fit``. So early stopping is wired on
  the constructor — on the search's base estimator and on the final estimator
  built in :meth:`DetectionClassifier._build_estimator` — whenever a monitor
  (validation) set is available (a plain fit with ``early_stopping_rounds`` set
  but no ``eval_set`` would raise).
* ``use_label_encoder`` was removed entirely: XGBoost no longer encodes targets,
  so the flag the design sketch mentioned is neither needed nor accepted here.
  Passing it only provokes an "unused parameter" warning; the labels are already
  ``{0, 1}``.

Hyperparameter search runs with ``refit=False`` and early stopping active inside
every fold: ``n_estimators`` is pinned at :data:`MAX_N_ESTIMATORS` and the number
of trees is chosen by early stopping against a held-out monitor rather than tuned
in the grid. Tuning the tree count *and* early-stopping on it makes the two
compete — whenever the search picks a small count, early stopping has no room to
act — so the count is left to early stopping alone. The search returns only the
shape parameters; :meth:`train` reads the selected tree count off the final fit.
Skipping the search's own refit avoids training a model the caller immediately
replaces.
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

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

#: RandomizedSearchCV sampling space (CLAUDE.md evaluation strategy). Note the
#: absence of ``n_estimators``: the tree count is not tuned. Searching it against
#: the same monitor early stopping consults makes the two compete — whenever the
#: search picks a small count, early stopping cannot fire and the monitor rows buy
#: no regularization (observed: a run stopped at ``best_iteration`` 197 against a
#: ceiling of 200, the ceiling binding, not the validation curve). Instead the
#: ceiling is fixed high (:data:`MAX_N_ESTIMATORS`) and early stopping selects the
#: real count on a held-out monitor.
SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

#: Ceiling for the boosting rounds. Never the operative count — early stopping
#: cuts training well below this — but high enough that early stopping, not the
#: ceiling, is what stops the fit. If a fit ever reaches it, raise it.
MAX_N_ESTIMATORS = 2000

#: Fraction of the training rows carved off as the early-stopping monitor when the
#: caller supplies no external validation set. The monitor only chooses the tree
#: count, after which the final model is refit on every training row (see
#: :meth:`DetectionClassifier.train`), so these rows are not withheld from it.
INTERNAL_VAL_FRACTION = 0.15

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
        tree *shape* parameters; otherwise :data:`DEFAULT_PARAMS` is used.

        The tree *count* is never tuned. ``n_estimators`` is pinned at the
        :data:`MAX_N_ESTIMATORS` ceiling and early stopping (patience
        :data:`EARLY_STOPPING_ROUNDS`) selects the real count on a held-out
        monitor — both inside every CV fold and on the final fit. This keeps the
        search and early stopping from competing for the same decision: with
        ``n_estimators`` in the grid the search could pick a count below where the
        validation curve plateaued, so early stopping never fired and the monitor
        rows bought no regularization.

        The monitor is ``(X_val, y_val)`` when supplied; otherwise a stratified
        :data:`INTERNAL_VAL_FRACTION` slice is carved from the training rows. In
        the internal case the monitor's only job is to choose the tree count, so
        the final model is **refit on every training row** with that fixed count —
        no rows are withheld from it. With an external monitor the fitted
        early-stopped model is kept as-is (the caller is holding the monitor out
        deliberately, e.g. the headline evaluation's validation split).

        Args:
            X_train: Training features, shape ``(n_samples, 16)``.
            y_train: Training labels in ``{0, 1}``, shape ``(n_samples,)``.
            X_val: Optional external validation features for early stopping.
            y_val: Optional external validation labels for early stopping.
            tune_hyperparameters: Whether to run the randomized search.

        Returns:
            A dict with ``best_params`` (the parameters used, with ``n_estimators``
            set to the count early stopping selected), ``cv_scores`` (the search's
            best score and spread, or ``{}`` when tuning is skipped), and
            ``training_time_seconds``.
        """
        features = np.asarray(X_train, dtype=np.float64)
        labels = np.asarray(y_train).astype(int)
        external_val = X_val is not None and y_val is not None
        # Early stopping is used whenever we tune (carving an internal monitor if
        # the caller gave none) or whenever an explicit validation set is handed in.
        use_early_stopping = external_val or tune_hyperparameters
        # An internally-carved monitor is disposable — the final model is refit on
        # the full training set once the count is known. An external one is not
        # ours to fold back in, so its early-stopped model is kept.
        refit_on_full = use_early_stopping and not external_val

        start = time.perf_counter()

        if not use_early_stopping:
            fit_features, fit_labels = features, labels
            es_features = es_labels = None
        elif external_val:
            fit_features, fit_labels = features, labels
            es_features = np.asarray(X_val, dtype=np.float64)
            es_labels = np.asarray(y_val).astype(int)
        else:
            fit_features, es_features, fit_labels, es_labels = train_test_split(
                features,
                labels,
                test_size=INTERNAL_VAL_FRACTION,
                stratify=labels,
                random_state=self.config.random_seed,
                shuffle=True,
            )

        if tune_hyperparameters:
            # Tuning implies early stopping, so the monitor is always set here;
            # guard rather than assert (asserts are stripped under -O).
            if es_features is None or es_labels is None:  # pragma: no cover
                raise ClassifierError("Early-stopping monitor missing while tuning")
            logger.info(
                "Tuning hyperparameters: RandomizedSearchCV "
                "(n_iter=%d, %d-fold, scoring=%r); early stopping selects n_estimators",
                SEARCH_N_ITER,
                SEARCH_CV_FOLDS,
                SEARCH_SCORING,
            )
            best_params, cv_scores = self._search_hyperparameters(
                fit_features, fit_labels, es_features, es_labels
            )
        else:
            logger.info("Training with default hyperparameters (tuning disabled)")
            best_params = dict(DEFAULT_PARAMS)
            cv_scores = {}

        # Ceiling for the early-stopping fit: the high cap for a tuned run (the
        # search returns only shape params), or DEFAULT_PARAMS' own count on a
        # --no-tune dry run.
        ceiling = int(best_params.get("n_estimators", MAX_N_ESTIMATORS))
        self.model = self._build_estimator(
            {**best_params, "n_estimators": ceiling},
            use_early_stopping=use_early_stopping,
        )
        if use_early_stopping:
            self.model.fit(
                fit_features,
                fit_labels,
                eval_set=[(es_features, es_labels)],
                verbose=False,
            )
        else:
            self.model.fit(fit_features, fit_labels)

        selected_trees: int | None = None
        best_iteration = getattr(self.model, "best_iteration", None)
        if use_early_stopping and best_iteration is not None:
            selected_trees = int(best_iteration) + 1

        if refit_on_full and selected_trees is not None:
            # The monitor has served its purpose (choosing the count); refit on
            # every training row with that fixed count so none are withheld from
            # the shipped model. No eval_set → no early stopping on the refit.
            self.model = self._build_estimator(
                {**best_params, "n_estimators": selected_trees},
                use_early_stopping=False,
            )
            self.model.fit(features, labels)

        if selected_trees is not None:
            # Report the count early stopping chose, not the ceiling.
            best_params = {**best_params, "n_estimators": selected_trees}

        self.is_fitted = True
        elapsed = time.perf_counter() - start

        self._log_training_summary(best_params, elapsed, selected_trees)
        return {
            "best_params": best_params,
            "cv_scores": cv_scores,
            "training_time_seconds": elapsed,
        }

    def _search_hyperparameters(
        self,
        X: FloatArray,
        y: IntArray,
        X_es: FloatArray,
        y_es: IntArray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the randomized CV search and return ``(best_params, cv_scores)``.

        ``n_estimators`` is not searched: the base estimator is pinned at the
        :data:`MAX_N_ESTIMATORS` ceiling with early stopping, so every fold trains
        against the shared held-out monitor ``(X_es, y_es)`` and stops at its own
        best iteration. ``best_params`` therefore carries only the shape
        parameters; the tree count is read off the final fit in :meth:`train`. The
        monitor is disjoint from ``X`` (the fold test slices are subsets of ``X``),
        so scoring stays honest — early stopping on the monitor cannot leak into a
        fold's ``f1``.

        Uses ``refit=False``: for a single scorer the best-parameter attributes
        are still populated, and skipping the refit avoids training a model the
        caller immediately replaces with the final fit.
        """
        base = xgb.XGBClassifier(
            random_state=self.config.random_seed,
            eval_metric="logloss",
            enable_categorical=False,
            n_estimators=MAX_N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
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
        # eval_set is a single-element list, so sklearn's fit-param handling leaves
        # it unindexed and hands the whole monitor to every fold — exactly the
        # shared early-stopping set we want (verified on sklearn 1.9 / xgboost 2.1).
        search.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)

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
        self, best_params: dict[str, Any], elapsed: float, selected_trees: int | None
    ) -> None:
        """Emit a one-line summary, noting the early-stopping tree count if any."""
        if selected_trees is not None:
            logger.info(
                "Training complete in %.2fs; early stopping selected %d trees",
                elapsed,
                selected_trees,
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

"""Tests for the AI detection inference pipeline.

The classifier and the feature pipeline are both stubbed, so the suite is fast
and offline: no language model is loaded and no trained model file is read. The
one exception is the end-to-end test, which is marked ``slow`` and gated on the
real CodeGen model being cached locally.

Stub wiring trick: the stub pipeline writes a file's intended AI probability into
the ``f1`` slot of its feature vector, and the echo classifier reads that slot
back as ``predict_proba``. That lets a test assign a distinct probability *per
file* and check the LOC-weighted commit average, without any real model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
import pytest

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ModelNotLoadedError
from deltx.detection.inference import AIDetectionInference
from deltx.detection.models import FeatureVector, FileAnalysisResult

FloatArray = npt.NDArray[np.float64]

_TS = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)

_REAL_SOURCE = '''def fibonacci(limit):
    """Yield Fibonacci numbers below limit."""
    previous, current = 0, 1
    while previous < limit:
        yield previous
        previous, current = current, previous + current
'''


class _FilePlan(TypedDict, total=False):
    """How the stub pipeline should render one file."""

    prob: float  # written into f1; the echo classifier reads it back
    loc: int
    parseable: bool
    error: NotRequired[str | None]


def _vector(f1: float = 0.0) -> FeatureVector:
    """A feature vector that is all zeros except ``f1`` (the probability channel)."""
    fields = dict.fromkeys(FeatureVector.feature_names(), 0.0)
    fields["f1_mean_surprisal"] = f1
    return FeatureVector(**fields)


class _StubPipeline:
    """Stands in for FeatureExtractionPipeline: pre-planned results, a call log."""

    def __init__(self, plan: dict[Path, _FilePlan]) -> None:
        self.plan = plan
        self.seen: list[Path] = []

    def extract_file_features(
        self, source_code: str, file_path: Path
    ) -> FileAnalysisResult:
        self.seen.append(file_path)
        spec = self.plan[file_path]
        return FileAnalysisResult(
            file_path=file_path,
            feature_vector=_vector(spec.get("prob", 0.0)),
            ai_confidence=0.0,  # placeholder; inference fills it in
            lines_of_code=spec.get("loc", 0),
            is_parseable=spec.get("parseable", True),
            error_message=spec.get("error"),
        )


class _EchoClassifier:
    """predict_proba echoes the f1 column, recovering each file's planned prob."""

    def predict_proba(self, features: FloatArray) -> FloatArray:
        return np.asarray(features, dtype=np.float64)[:, 0]


class _FixedClassifier:
    """predict_proba returns a constant for every row, and counts its calls."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def predict_proba(self, features: FloatArray) -> FloatArray:
        self.calls += 1
        rows = np.asarray(features, dtype=np.float64).shape[0]
        return np.full(rows, self.value, dtype=np.float64)


class _CrashingPipeline:
    """A pipeline that raises outright, exercising analyze_file's crash guard."""

    def extract_file_features(
        self, source_code: str, file_path: Path
    ) -> FileAnalysisResult:
        raise RuntimeError("pipeline exploded")


def _detector(pipeline: object, classifier: object) -> AIDetectionInference:
    """Build an inference pipeline from stub components (typed loosely on purpose)."""
    return AIDetectionInference(pipeline, classifier)  # type: ignore[arg-type]


# -- 1. analyze_file populates confidence from the classifier -----------------


def test_analyze_file_populates_confidence_from_classifier() -> None:
    """A clean file is classified and its probability lands on ai_confidence."""
    path = Path("module.py")
    pipeline = _StubPipeline({path: {"loc": 10, "parseable": True}})
    classifier = _FixedClassifier(0.85)
    detector = _detector(pipeline, classifier)

    result = detector.analyze_file("value = 1\n", path)

    assert result.ai_confidence == pytest.approx(0.85)
    assert result.is_parseable
    assert classifier.calls == 1


def test_analyze_file_survives_pipeline_crash() -> None:
    """A pipeline that raises is contained: the file is returned, not the error."""
    classifier = _FixedClassifier(0.9)
    detector = _detector(_CrashingPipeline(), classifier)

    result = detector.analyze_file("value = 1\n", Path("boom.py"))

    assert result.ai_confidence == 0.0
    assert result.is_parseable is False
    assert result.error_message is not None
    assert "extraction error" in result.error_message
    assert classifier.calls == 0  # a crashed file is never classified


# -- 2. commit-level LOC-weighted average -------------------------------------


def test_analyze_commit_loc_weighted_average() -> None:
    """ai_confidence_pct is the LOC-weighted mean of per-file probabilities."""
    file_a, file_b, file_c = Path("a.py"), Path("b.py"), Path("c.py")
    plan: dict[Path, _FilePlan] = {
        file_a: {"prob": 0.9, "loc": 100},
        file_b: {"prob": 0.1, "loc": 200},
        file_c: {"prob": 0.5, "loc": 50},
    }
    detector = _detector(_StubPipeline(plan), _EchoClassifier())
    files = {file_a: "a", file_b: "b", file_c: "c"}

    result = detector.analyze_commit(files, "deadbeefcafe1234", _TS)

    # (0.9*100 + 0.1*200 + 0.5*50) / 350 * 100 = 38.57% — note the task's "39.3%"
    # is an arithmetic slip; the correct weighted mean is asserted here.
    expected = (0.9 * 100 + 0.1 * 200 + 0.5 * 50) / 350 * 100
    assert result.ai_confidence_pct == pytest.approx(expected)
    assert result.ai_confidence_pct == pytest.approx(38.5714, abs=1e-3)
    assert result.total_files_analyzed == 3
    assert result.total_files_skipped == 0


# -- 3. non-Python files are filtered out -------------------------------------


def test_analyze_commit_skips_non_python_files() -> None:
    """A README and a JSON file never reach the pipeline or the results."""
    py = Path("main.py")
    pipeline = _StubPipeline({py: {"prob": 0.7, "loc": 20}})
    detector = _detector(pipeline, _EchoClassifier())
    files = {
        py: "value = 1\n",
        Path("readme.md"): "# hello",
        Path("data.json"): "{}",
    }

    result = detector.analyze_commit(files, "abc12345", _TS)

    assert pipeline.seen == [py]  # only the .py file was analyzed
    assert [r.file_path for r in result.file_results] == [py]
    assert result.total_files_analyzed == 1
    assert result.total_files_skipped == 2
    assert result.ai_confidence_pct == pytest.approx(70.0)


@pytest.mark.parametrize(
    "skipped",
    ["setup.py", "conftest.py", "__pycache__/mod.py", "pkg/__pycache__/x.py"],
)
def test_analyze_commit_skips_tooling_and_pycache(skipped: str) -> None:
    """Packaging/test-config files and __pycache__ output are excluded."""
    keep = Path("pkg/core.py")
    pipeline = _StubPipeline({keep: {"prob": 0.6, "loc": 15}})
    detector = _detector(pipeline, _EchoClassifier())
    files = {keep: "value = 1\n", Path(skipped): "value = 1\n"}

    result = detector.analyze_commit(files, "hash0001", _TS)

    assert pipeline.seen == [keep]
    assert result.total_files_skipped == 1


# -- 4. all-unparseable commit scores zero ------------------------------------


def test_analyze_commit_all_unparseable_is_zero() -> None:
    """When no file is classifiable, the commit scores 0.0 (assume human)."""
    file_a, file_b = Path("a.py"), Path("b.py")
    plan: dict[Path, _FilePlan] = {
        file_a: {"loc": 0, "parseable": False, "error": "SyntaxError"},
        file_b: {"loc": 0, "parseable": False, "error": "SyntaxError"},
    }
    detector = _detector(_StubPipeline(plan), _EchoClassifier())

    result = detector.analyze_commit(
        {file_a: "def (:", file_b: "!!!"}, "bad00000", _TS
    )

    assert result.ai_confidence_pct == 0.0
    assert result.total_files_analyzed == 2
    assert all(not r.is_parseable for r in result.file_results)


# -- 5. empty commit scores zero ----------------------------------------------


def test_analyze_commit_empty_files_is_zero() -> None:
    detector = _detector(_StubPipeline({}), _EchoClassifier())

    result = detector.analyze_commit({}, "empty000", _TS)

    assert result.ai_confidence_pct == 0.0
    assert result.total_files_analyzed == 0
    assert result.total_files_skipped == 0
    assert result.file_results == []


# -- 6. batch processing ------------------------------------------------------


def test_analyze_commit_batch_processes_each_commit() -> None:
    """Each commit is analyzed independently, order and author preserved."""
    file_a, file_b = Path("a.py"), Path("b.py")
    plan: dict[Path, _FilePlan] = {
        file_a: {"prob": 0.8, "loc": 100},
        file_b: {"prob": 0.2, "loc": 100},
    }
    detector = _detector(_StubPipeline(plan), _EchoClassifier())
    commits = [
        {"files": {file_a: "a"}, "commit_hash": "c0ffee00", "timestamp": _TS},
        {
            "files": {file_b: "b"},
            "commit_hash": "deadbeef",
            "timestamp": _TS,
            "author": "alice",
        },
    ]

    results = detector.analyze_commit_batch(commits, progress=False)

    assert [r.commit_hash for r in results] == ["c0ffee00", "deadbeef"]
    assert results[0].ai_confidence_pct == pytest.approx(80.0)
    assert results[1].ai_confidence_pct == pytest.approx(20.0)
    assert results[0].author is None
    assert results[1].author == "alice"


def test_analyze_commit_batch_with_progress_bar_runs() -> None:
    """The progress=True branch renders without error and returns results."""
    file_a = Path("a.py")
    detector = _detector(
        _StubPipeline({file_a: {"prob": 0.5, "loc": 10}}), _EchoClassifier()
    )
    commits = [{"files": {file_a: "a"}, "commit_hash": "aaaa1111", "timestamp": _TS}]

    results = detector.analyze_commit_batch(commits, progress=True)

    assert len(results) == 1
    assert results[0].ai_confidence_pct == pytest.approx(50.0)


# -- 7. from_config guards a missing classifier -------------------------------


def test_from_config_raises_when_classifier_missing(
    config: DeltxConfig, tmp_path: Path
) -> None:
    """No trained model on disk → ModelNotLoadedError, not a silent empty model."""
    config.classifier_path = tmp_path / "absent.joblib"

    with pytest.raises(ModelNotLoadedError):
        AIDetectionInference.from_config(config)


def test_from_config_loads_saved_classifier(
    config: DeltxConfig, tmp_path: Path
) -> None:
    """A classifier saved to config.classifier_path is loaded into a ready detector."""
    from deltx.detection.classifier import DetectionClassifier

    config.classifier_path = tmp_path / "detector.joblib"
    rng = np.random.default_rng(1)
    features = rng.normal(size=(20, 16))
    labels = np.array([0, 1] * 10)
    trained = DetectionClassifier(config)
    trained.train(features, labels, tune_hyperparameters=False)
    trained.save(config.classifier_path)

    detector = AIDetectionInference.from_config(config)

    assert isinstance(detector, AIDetectionInference)
    assert detector.classifier.is_fitted


# -- extra: unclassifiable files are excluded from the average ----------------


def test_analyze_file_family_failure_is_not_classified() -> None:
    """A file that parsed but had a feature family fail is not sent to the model."""
    path = Path("weird.py")
    plan: dict[Path, _FilePlan] = {
        path: {"prob": 0.99, "loc": 12, "parseable": True, "error": "perplexity: boom"}
    }
    classifier = _FixedClassifier(0.99)
    detector = _detector(_StubPipeline(plan), classifier)

    result = detector.analyze_file("value = 1\n", path)

    assert result.ai_confidence == 0.0
    assert result.is_parseable is False  # forced, so the commit average skips it
    assert classifier.calls == 0  # never classified


def test_analyze_commit_excludes_family_failures_from_average() -> None:
    """A parsed-but-failed file must not drag the commit toward a fake 0.0."""
    good, bad = Path("good.py"), Path("bad.py")
    plan: dict[Path, _FilePlan] = {
        good: {"prob": 0.4, "loc": 100, "parseable": True},
        bad: {"prob": 0.99, "loc": 100, "parseable": True, "error": "dist: boom"},
    }
    detector = _detector(_StubPipeline(plan), _EchoClassifier())

    result = detector.analyze_commit({good: "g", bad: "b"}, "hash0002", _TS)

    # Only `good` counts: 0.4 * 100 / 100 * 100 = 40.0; `bad` is excluded.
    assert result.ai_confidence_pct == pytest.approx(40.0)
    assert result.total_files_analyzed == 2


# -- 8. end-to-end with the real model (slow) ---------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("require_model")
def test_end_to_end_real_file(config: DeltxConfig) -> None:
    """Real feature extraction + a throwaway trained classifier over one file."""
    from deltx.detection.classifier import DetectionClassifier
    from deltx.detection.pipeline import FeatureExtractionPipeline

    # A tiny classifier trained on synthetic 16-D data — no saved model needed.
    rng = np.random.default_rng(0)
    features = np.vstack(
        [rng.normal(0.0, 1.0, (60, 16)), rng.normal(1.5, 1.0, (60, 16))]
    )
    labels = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
    classifier = DetectionClassifier(config)
    classifier.train(features, labels, tune_hyperparameters=False)

    detector = AIDetectionInference(FeatureExtractionPipeline(config), classifier)
    result = detector.analyze_file(_REAL_SOURCE, Path("sample.py"))

    assert result.is_parseable
    assert result.lines_of_code > 0
    assert 0.0 <= result.ai_confidence <= 1.0

    commit = detector.analyze_commit({Path("sample.py"): _REAL_SOURCE}, "abc123", _TS)
    assert 0.0 <= commit.ai_confidence_pct <= 100.0

"""Tests for the detection CLI (``analyze`` and ``analyze-dir``).

The detector is stubbed at :func:`AIDetectionInference.from_config`, so the CLI's
own wiring — argument parsing, file reading, JSON rendering, the summary table,
and the missing-classifier error path — is exercised without loading a language
model or a trained classifier. Only real result models cross the boundary, so the
serialisation the CLI performs is genuine.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ModelNotLoadedError
from deltx.detection import cli as cli_module
from deltx.detection.models import (
    CommitAnalysisResult,
    FeatureVector,
    FileAnalysisResult,
)


def _vector() -> FeatureVector:
    return FeatureVector(**dict.fromkeys(FeatureVector.feature_names(), 0.0))


class _StubDetector:
    """A detector that returns fixed result models without any model loading."""

    def analyze_file(
        self, source_code: str, file_path: Path
    ) -> FileAnalysisResult:
        return FileAnalysisResult(
            file_path=file_path,
            feature_vector=_vector(),
            ai_confidence=0.73,
            lines_of_code=5,
            is_parseable=True,
        )

    def analyze_commit(
        self, files: dict[Path, str], commit_hash: str, timestamp: datetime
    ) -> CommitAnalysisResult:
        results = [
            FileAnalysisResult(
                file_path=path,
                feature_vector=_vector(),
                ai_confidence=0.5,
                lines_of_code=3,
                is_parseable=True,
            )
            for path in files
        ]
        return CommitAnalysisResult(
            commit_hash=commit_hash,
            timestamp=timestamp,
            ai_confidence_pct=50.0,
            file_results=results,
            total_files_analyzed=len(results),
            total_files_skipped=0,
        )


@pytest.fixture
def stub_detector(monkeypatch: pytest.MonkeyPatch) -> _StubDetector:
    """Make ``from_config`` return a stub, so _build_detector runs for real."""
    detector = _StubDetector()
    monkeypatch.setattr(
        cli_module.AIDetectionInference,
        "from_config",
        lambda config: detector,
    )
    return detector


def test_analyze_prints_json(stub_detector: _StubDetector, tmp_path: Path) -> None:
    py_file = tmp_path / "sample.py"
    py_file.write_text("value = 1\n", encoding="utf-8")

    result = CliRunner().invoke(cli_module.cli, ["analyze", "--file", str(py_file)])

    assert result.exit_code == 0, result.output
    assert "ai_confidence" in result.output
    assert "0.73" in result.output


def test_analyze_dir_prints_summary(
    stub_detector: _StubDetector, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("value = 2\n", encoding="utf-8")

    result = CliRunner().invoke(cli_module.cli, ["analyze-dir", "--dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "files analyzed" in result.output
    assert "50.0" in result.output


def test_analyze_dir_no_python_files(
    stub_detector: _StubDetector, tmp_path: Path
) -> None:
    result = CliRunner().invoke(cli_module.cli, ["analyze-dir", "--dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "No Python files found" in result.output


def test_analyze_missing_classifier_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise(config: DeltxConfig) -> _StubDetector:
        raise ModelNotLoadedError("No classifier file at data/models/detector.joblib")

    monkeypatch.setattr(cli_module.AIDetectionInference, "from_config", _raise)
    py_file = tmp_path / "sample.py"
    py_file.write_text("value = 1\n", encoding="utf-8")

    result = CliRunner().invoke(cli_module.cli, ["analyze", "--file", str(py_file)])

    assert result.exit_code == 1
    assert "Cannot start detector" in result.output

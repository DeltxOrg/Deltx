"""Tests for the deltx-detect command line interface."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from deltx.common.exceptions import CheckpointError
from deltx.detection import cli as cli_module
from deltx.detection.cli import cli
from deltx.detection.detector import ScoredSource
from deltx.detection.inference import AIDetectionInference
from deltx.detection.models import ClassDistribution

SOURCE = "def f():\n    return 1\n"


class StubDetector:
    def score_source(self, source: str) -> ScoredSource:
        return ScoredSource(
            distribution=ClassDistribution.from_probabilities([0.25, 0.75]),
            token_count=10,
            chunk_count=1,
        )


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """A runner with model loading stubbed out."""
    monkeypatch.setattr(
        AIDetectionInference,
        "from_config",
        classmethod(lambda cls, config=None: cls(StubDetector())),
    )
    return CliRunner()


class TestAnalyze:
    def test_emits_json_with_a_percentage(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        target = tmp_path / "m.py"
        target.write_text(SOURCE, encoding="utf-8")

        result = runner.invoke(cli, ["analyze", "--file", str(target)])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert payload["ai_confidence_pct"] == pytest.approx(75.0)
        assert payload["is_scored"] is True

    def test_missing_file_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", "--file", "nope.py"])
        assert result.exit_code != 0

    def test_checkpoint_error_becomes_a_clean_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing checkpoint must not surface as a traceback."""
        target = tmp_path / "m.py"
        target.write_text(SOURCE, encoding="utf-8")

        def boom(cls: object, config: object = None) -> None:
            raise CheckpointError("no checkpoint")

        monkeypatch.setattr(AIDetectionInference, "from_config", classmethod(boom))
        result = CliRunner().invoke(cli, ["analyze", "--file", str(target)])
        assert result.exit_code != 0
        assert "no checkpoint" in result.output


class TestAnalyzeDir:
    def test_reports_files_and_commit_score(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text(SOURCE, encoding="utf-8")
        (tmp_path / "b.py").write_text(SOURCE, encoding="utf-8")

        result = runner.invoke(cli, ["analyze-dir", "--dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "ai_confidence_pct" in result.output
        assert "75.00" in result.output

    def test_directory_without_python_is_rejected(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        (tmp_path / "README.md").write_text("# docs", encoding="utf-8")
        result = runner.invoke(cli, ["analyze-dir", "--dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "no analysable Python files" in result.output

    def test_boilerplate_is_excluded_from_candidates(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        (tmp_path / "setup.py").write_text(SOURCE, encoding="utf-8")
        result = runner.invoke(cli, ["analyze-dir", "--dir", str(tmp_path)])
        assert result.exit_code != 0


def test_result_row_renders_skipped_files() -> None:
    from deltx.detection.models import FileAnalysisResult

    row = cli_module._result_row(
        FileAnalysisResult.unscored(Path("a.txt"), "not a Python file")
    )
    assert row[1] == "—"
    assert row[2] == "skipped"

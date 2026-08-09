"""End-to-end CLI tests with a mocked detector.

A real local Git repository is cloned by the CLI itself; only the DroidDetect
checkpoint is faked, so the clone, traversal, aggregation, Parquet write, and
resume paths all run for real without a download.
"""

from collections.abc import Callable

import pandas as pd
import pytest
from click.testing import CliRunner

from deltx.common.constants import EXTRACTION_COLUMNS
from deltx.detection.inference import AIDetectionInference
from deltx.extraction import cli

from .conftest import AI_MARKER, GitRepoBuilder

Builder = Callable[[str], GitRepoBuilder]


@pytest.fixture
def patched_detector(
    monkeypatch: pytest.MonkeyPatch, fake_inference: AIDetectionInference
) -> None:
    """Replace the real detector loader with the fake inference facade."""
    monkeypatch.setattr(
        cli.AIDetectionInference,
        "from_config",
        lambda config: fake_inference,
    )


def _source_repo(builder: GitRepoBuilder) -> GitRepoBuilder:
    builder.write("human.py", "def a():\n    return 1\n")
    builder.commit("first: human code")
    builder.write("ai.py", f"{AI_MARKER}\ndef b():\n    return 2\n")
    builder.commit("second: model code")
    builder.write("NOTES.md", "docs\n")
    builder.commit("third: docs only")
    return builder


def test_end_to_end_produces_one_row_per_commit(
    repo_builder: Builder, patched_detector: None, tmp_path: pytest.TempPathFactory
) -> None:
    source = _source_repo(repo_builder("source"))
    output = source.root.parent / "out" / "result.parquet"

    result = CliRunner().invoke(
        cli.main,
        [
            "--repo-url",
            str(source.root),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.with_suffix(".log").exists()

    frame = pd.read_parquet(output)
    assert list(frame.columns) == list(EXTRACTION_COLUMNS)
    assert len(frame) == 3
    assert list(frame["commit_index"]) == [0, 1, 2]
    assert list(frame["commit_message"]) == [
        "first: human code",
        "second: model code",
        "third: docs only",
    ]
    # Human commit low, AI commit high, docs-only commit NaN.
    assert frame.loc[0, "ai_confidence_pct"] < 20.0
    assert frame.loc[1, "ai_confidence_pct"] > 80.0
    assert pd.isna(frame.loc[2, "ai_confidence_pct"])
    assert frame.loc[2, "files_changed_py"] == 0


def test_resume_appends_only_new_commits(
    repo_builder: Builder, patched_detector: None
) -> None:
    source = _source_repo(repo_builder("source"))
    output = source.root.parent / "result.parquet"
    runner = CliRunner()

    first = runner.invoke(
        cli.main,
        ["--repo-url", str(source.root), "--output", str(output), "--device", "cpu"],
    )
    assert first.exit_code == 0, first.output
    before = pd.read_parquet(output)
    assert len(before) == 3

    # A new commit lands upstream; resume should add exactly that one.
    source.write("more.py", "c = 3\n")
    source.commit("fourth: more code")

    second = runner.invoke(
        cli.main,
        [
            "--repo-url",
            str(source.root),
            "--output",
            str(output),
            "--resume",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert second.exit_code == 0, second.output

    after = pd.read_parquet(output)
    assert len(after) == 4
    assert list(after["commit_index"]) == [0, 1, 2, 3]
    assert after.loc[3, "commit_message"] == "fourth: more code"
    # Earlier rows are unchanged, not recomputed into different values.
    pd.testing.assert_frame_equal(before, after.iloc[:3])


def test_unknown_branch_fails_cleanly(
    repo_builder: Builder, patched_detector: None
) -> None:
    source = _source_repo(repo_builder("source"))
    output = source.root.parent / "result.parquet"

    result = CliRunner().invoke(
        cli.main,
        [
            "--repo-url",
            str(source.root),
            "--output",
            str(output),
            "--branch",
            "nonexistent",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()

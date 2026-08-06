"""Tests for the per-commit extraction pipeline."""

import json
import math
from collections.abc import Callable

from deltx.detection.inference import AIDetectionInference
from deltx.extraction.git_history import GitRepository
from deltx.extraction.pipeline import CommitAiConfidenceExtractor

from .conftest import AI_MARKER, GitRepoBuilder

Builder = Callable[[str], GitRepoBuilder]
_URL = "https://example.com/r.git"


def _extractor(
    builder: GitRepoBuilder, inference: AIDetectionInference
) -> tuple[CommitAiConfidenceExtractor, GitRepository]:
    repo = GitRepository(builder.root)
    return CommitAiConfidenceExtractor(inference, repo, _URL), repo


def test_commit_with_no_python_is_empty_nan(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    b.write("README.md", "hello\n")
    b.commit("docs only")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 0)

    assert row.files_changed_py == 0
    assert row.total_loc_scored == 0
    assert math.isnan(row.ai_confidence_pct)
    assert row.file_scores_json == "{}"
    assert row.repo_url == _URL


def test_human_code_scores_low(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    b.write("a.py", "def a():\n    return 1\n")
    b.commit("human code")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 0)

    assert row.files_changed_py == 1
    assert row.total_loc_scored == 2
    assert row.ai_confidence_pct < 20.0  # P(AI)=0.1 -> 10%
    scores = json.loads(row.file_scores_json)
    assert set(scores) == {"a.py"}
    assert scores["a.py"]["loc"] == 2


def test_ai_code_scores_high(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    b.write("a.py", f"{AI_MARKER}\ndef a():\n    return 1\n")
    b.commit("model code")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 0)
    assert row.ai_confidence_pct > 80.0  # P(AI)=0.9 -> 90%


def test_loc_weighted_average_across_files(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    # human.py: 1 LOC, P(AI)=0.1. ai.py: the marker is a comment (not counted
    # as LOC), leaving 2 code lines at P(AI)=0.9.
    b.write("human.py", "x = 1\n")
    b.write("ai.py", f"{AI_MARKER}\ny = 2\nz = 3\n")
    b.commit("mixed")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 0)

    # (0.1*1 + 0.9*2) / (1+2) * 100 = 63.333...
    assert row.total_loc_scored == 3
    assert math.isclose(row.ai_confidence_pct, 63.3333, abs_tol=0.01)


def test_empty_file_counted_but_not_scored(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    b.write("real.py", "x = 1\n")
    b.write("empty.py", "")
    b.commit("with empty")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 0)

    assert row.files_changed_py == 2  # both are added .py files
    assert row.total_loc_scored == 1  # only real.py contributed
    assert set(json.loads(row.file_scores_json)) == {"real.py"}


def test_binary_file_skipped_leaves_nan(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    b.write_bytes("weird.py", b"\x00\x01not text\x00")
    b.commit("binary py")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 0)

    assert row.files_changed_py == 1  # git saw a changed .py
    assert row.total_loc_scored == 0  # but nothing decodable to score
    assert math.isnan(row.ai_confidence_pct)


def test_commit_index_is_recorded(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 1\n")
    b.commit("first")
    extractor, repo = _extractor(b, fake_inference)

    row = extractor.build_row(repo.iter_commits("main")[0], 5)
    assert row.commit_index == 5

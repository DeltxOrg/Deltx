"""Tests for the extraction row model."""

import json
import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deltx.extraction.models import CommitRow

_TS = datetime(2024, 1, 1, tzinfo=UTC)


def _row(**overrides: object) -> CommitRow:
    base: dict[str, object] = {
        "repo_url": "https://example.com/r.git",
        "commit_hash": "abc123",
        "commit_timestamp": _TS,
        "commit_author": "Ann",
        "commit_message": "msg",
        "commit_index": 0,
        "files_changed_py": 1,
        "total_loc_scored": 10,
        "ai_confidence_pct": 42.0,
    }
    base.update(overrides)
    return CommitRow(**base)  # type: ignore[arg-type]


def test_valid_row_round_trips() -> None:
    row = _row()
    assert row.ai_confidence_pct == 42.0
    assert row.file_scores_json == "{}"


def test_nan_confidence_is_allowed() -> None:
    row = _row(ai_confidence_pct=math.nan)
    assert math.isnan(row.ai_confidence_pct)


@pytest.mark.parametrize("bad", [-0.1, 100.1, 250.0])
def test_out_of_range_confidence_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        _row(ai_confidence_pct=bad)


def test_empty_builder_sets_nan_and_zero_counts() -> None:
    row = CommitRow.empty(
        repo_url="u",
        commit_hash="h",
        commit_timestamp=_TS,
        commit_author="a",
        commit_message="m",
        commit_index=7,
    )
    assert row.files_changed_py == 0
    assert row.total_loc_scored == 0
    assert math.isnan(row.ai_confidence_pct)
    assert row.file_scores_json == "{}"
    assert row.commit_index == 7


def test_encode_file_scores_is_sorted_and_compact() -> None:
    encoded = CommitRow.encode_file_scores(
        {"b.py": {"score": 0.9, "loc": 5}, "a.py": {"score": 0.1, "loc": 3}}
    )
    # Deterministic key order regardless of insertion order.
    assert list(json.loads(encoded)) == ["a.py", "b.py"]
    assert " " not in encoded

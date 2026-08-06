"""Tests for Parquet persistence and resume."""

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deltx.common.constants import EXTRACTION_COLUMNS
from deltx.common.exceptions import ExtractionError
from deltx.extraction.models import CommitRow
from deltx.extraction.storage import load_resume, rows_to_frame, write_parquet

_TS = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _row(index: int, sha: str, pct: float, changed: int = 1) -> CommitRow:
    return CommitRow(
        repo_url="https://example.com/r.git",
        commit_hash=sha,
        commit_timestamp=_TS,
        commit_author="Ann",
        commit_message=f"commit {index}",
        commit_index=index,
        files_changed_py=changed,
        total_loc_scored=10 if changed else 0,
        ai_confidence_pct=pct,
    )


def test_frame_has_canonical_columns_and_is_sorted() -> None:
    rows = [_row(2, "c", 30.0), _row(0, "a", 10.0), _row(1, "b", 20.0)]
    frame = rows_to_frame(rows)

    assert list(frame.columns) == list(EXTRACTION_COLUMNS)
    assert list(frame["commit_index"]) == [0, 1, 2]
    assert str(frame["commit_timestamp"].dtype) == "datetime64[ns, UTC]"
    assert frame["ai_confidence_pct"].dtype == "float64"


def test_write_and_reload_round_trip(tmp_path: Path) -> None:
    rows = [_row(0, "a", 10.0), _row(1, "b", 90.0)]
    out = tmp_path / "nested" / "r.parquet"
    write_parquet(rows, out)

    assert out.exists()
    reloaded, processed = load_resume(out)
    assert processed == {"a", "b"}
    assert [r.commit_index for r in reloaded] == [0, 1]
    assert reloaded[1].ai_confidence_pct == 90.0


def test_nan_confidence_survives_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    write_parquet([_row(0, "a", math.nan, changed=0)], out)

    reloaded, _ = load_resume(out)
    assert math.isnan(reloaded[0].ai_confidence_pct)


def test_write_is_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    write_parquet([_row(0, "a", 10.0)], out)
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_empty_rows_write_readable_file(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    write_parquet([], out)
    frame = pd.read_parquet(out)
    assert list(frame.columns) == list(EXTRACTION_COLUMNS)
    assert frame.empty


def test_load_resume_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        load_resume(tmp_path / "nope.parquet")


def test_load_resume_rejects_wrong_schema(tmp_path: Path) -> None:
    out = tmp_path / "bad.parquet"
    pd.DataFrame({"unexpected": [1, 2]}).to_parquet(out)
    with pytest.raises(ExtractionError):
        load_resume(out)

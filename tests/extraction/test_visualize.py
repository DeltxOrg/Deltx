"""Tests for the results-visualisation module.

Rendering runs on the headless Agg backend, so these need no display. They
assert that figures are produced and that summary statistics are correct;
pixel-level appearance is out of scope.
"""

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deltx.common.exceptions import ExtractionError
from deltx.extraction.models import CommitRow
from deltx.extraction.storage import write_parquet
from deltx.extraction.visualize import (
    load_results,
    render_report,
    summarise,
)

_TS = datetime(2024, 1, 1, tzinfo=UTC)


def _rows() -> list[CommitRow]:
    rows: list[CommitRow] = []
    for i in range(12):
        if i % 3 == 0:  # every third commit changed no Python
            rows.append(
                CommitRow.empty(
                    repo_url="https://example.com/demo.git",
                    commit_hash=f"h{i}",
                    commit_timestamp=_TS,
                    commit_author="Ann" if i % 2 else "Bob",
                    commit_message=f"commit {i}",
                    commit_index=i,
                )
            )
        else:
            rows.append(
                CommitRow(
                    repo_url="https://example.com/demo.git",
                    commit_hash=f"h{i}",
                    commit_timestamp=_TS,
                    commit_author="Ann" if i % 2 else "Bob",
                    commit_message=f"commit {i}",
                    commit_index=i,
                    files_changed_py=2,
                    total_loc_scored=40,
                    ai_confidence_pct=float(i * 7 % 100),
                )
            )
    return rows


@pytest.fixture
def results_parquet(tmp_path: Path) -> Path:
    out = tmp_path / "demo.parquet"
    write_parquet(_rows(), out)
    return out


def test_load_results_validates_and_sorts(results_parquet: Path) -> None:
    frame = load_results(results_parquet)
    assert list(frame["commit_index"]) == list(range(12))


def test_load_results_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        load_results(tmp_path / "nope.parquet")


def test_load_results_rejects_foreign_parquet(tmp_path: Path) -> None:
    out = tmp_path / "foreign.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(out)
    with pytest.raises(ExtractionError):
        load_results(out)


def test_summarise_counts(results_parquet: Path) -> None:
    frame = load_results(results_parquet)
    summary = summarise(frame, "demo")

    assert summary.total_commits == 12
    assert summary.scored_commits == 8
    assert summary.unscored_commits == 4
    assert summary.authors == 2
    assert not math.isnan(summary.mean_ai_pct)


def test_render_report_writes_all_figures(
    results_parquet: Path, tmp_path: Path
) -> None:
    frame = load_results(results_parquet)
    out_dir = tmp_path / "charts"
    written = render_report(frame, out_dir, repo_label="demo")

    assert len(written) == 5
    names = {p.name for p in written}
    assert names == {
        "demo_timeline.png",
        "demo_distribution.png",
        "demo_by_author.png",
        "demo_activity.png",
        "demo_dashboard.png",
    }
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_report_handles_all_unscored(tmp_path: Path) -> None:
    rows = [
        CommitRow.empty(
            repo_url="https://example.com/demo.git",
            commit_hash=f"h{i}",
            commit_timestamp=_TS,
            commit_author="Ann",
            commit_message=f"c{i}",
            commit_index=i,
        )
        for i in range(4)
    ]
    parquet = tmp_path / "empty.parquet"
    write_parquet(rows, parquet)
    frame = load_results(parquet)

    # Nothing scored: figures must still render without raising.
    written = render_report(frame, tmp_path / "charts", repo_label="empty")
    assert all(p.exists() for p in written)

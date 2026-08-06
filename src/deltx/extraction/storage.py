"""Parquet persistence and resume support for extraction output.

The output is a single Parquet file whose column order and dtypes are fixed by
:data:`deltx.common.constants.EXTRACTION_COLUMNS`. Writes are atomic and the
whole frame is rewritten on each checkpoint, so an interrupted run leaves a
valid file to resume from rather than a half-written one.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from deltx.common.constants import EXTRACTION_COLUMNS
from deltx.common.exceptions import ExtractionError
from deltx.extraction.models import CommitRow

logger = logging.getLogger(__name__)

#: Column dtypes for the output frame. Kept explicit so an all-NaN
#: ``ai_confidence_pct`` column (a run where nothing scored) still lands as
#: float, and an empty result still writes a well-typed, readable file.
_COLUMN_DTYPES: dict[str, str] = {
    "repo_url": "string",
    "commit_hash": "string",
    "commit_author": "string",
    "commit_message": "string",
    "commit_index": "int64",
    "files_changed_py": "int64",
    "total_loc_scored": "int64",
    "ai_confidence_pct": "float64",
    "file_scores_json": "string",
}


def rows_to_frame(rows: list[CommitRow]) -> pd.DataFrame:
    """Build a correctly-typed, column-ordered DataFrame from rows.

    Args:
        rows: Extraction rows, in any order.

    Returns:
        A DataFrame with the canonical column order and dtypes, sorted by
        ``commit_index`` so the series is chronological regardless of the order
        rows were produced or merged in.
    """
    frame = pd.DataFrame(
        [row.model_dump() for row in rows], columns=list(EXTRACTION_COLUMNS)
    )
    if not frame.empty:
        frame = frame.astype(_COLUMN_DTYPES)
        frame["commit_timestamp"] = pd.to_datetime(
            frame["commit_timestamp"], utc=True
        )
        frame = frame.sort_values("commit_index").reset_index(drop=True)
    return frame


def write_parquet(rows: list[CommitRow], output: Path) -> None:
    """Atomically write rows to ``output`` as Parquet.

    The frame is written to a sibling temporary file and then moved into place,
    so a crash mid-write cannot corrupt an existing output.

    Args:
        rows: Rows to persist.
        output: Destination ``.parquet`` path.

    Raises:
        ExtractionError: If the frame cannot be written.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = rows_to_frame(rows)
    tmp = output.with_suffix(output.suffix + ".tmp")
    try:
        frame.to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(tmp, output)
    except (OSError, ValueError, ImportError) as exc:
        tmp.unlink(missing_ok=True)
        msg = f"could not write Parquet to {output}: {exc}"
        raise ExtractionError(msg) from exc


def load_resume(resume_path: Path) -> tuple[list[CommitRow], set[str]]:
    """Load already-computed rows from a partial Parquet to resume from.

    Args:
        resume_path: Path to an existing partial output.

    Returns:
        A tuple of the previously-computed rows and the set of commit hashes
        they cover, so the pipeline can skip re-scoring them.

    Raises:
        ExtractionError: If the file is missing or unreadable.
    """
    if not resume_path.exists():
        msg = f"resume file not found: {resume_path}"
        raise ExtractionError(msg)
    try:
        frame = pd.read_parquet(resume_path, engine="pyarrow")
    except (OSError, ValueError, ImportError) as exc:
        msg = f"could not read resume file {resume_path}: {exc}"
        raise ExtractionError(msg) from exc

    missing = set(EXTRACTION_COLUMNS) - set(frame.columns)
    if missing:
        msg = f"resume file {resume_path} is missing columns: {sorted(missing)}"
        raise ExtractionError(msg)

    rows = [
        CommitRow.model_validate(record)
        for record in frame.to_dict(orient="records")
    ]
    processed = {row.commit_hash for row in rows}
    logger.info(
        "resuming from %s: %d commits already processed", resume_path, len(processed)
    )
    return rows, processed

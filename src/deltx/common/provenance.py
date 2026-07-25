"""Run provenance capture: what produced a result, and can it be reproduced.

A console scrollback is not a record. This module snapshots everything needed to
reproduce — or later cite — a training run: the code state it executed against,
the environment it ran in, a fingerprint of the exact data it consumed, and the
metrics it produced. The result is a JSON manifest written beside the
human-readable console report.

Three fields carry most of the weight and are easy to underestimate:

- ``DatasetFingerprint.sha256`` pins the *bytes* of the feature matrix. A file at
  a stable path is not a stable input; rebuilding or repairing it silently
  invalidates every number previously attributed to it.
- ``GitState.dirty``/``diff`` records uncommitted work. Citing a commit hash for a
  run that executed against modified files is simply false, and runs during active
  development are almost always dirty.
- ``Provenance.packages`` pins the libraries. XGBoost and scikit-learn change
  defaults across releases, so the same code and data can drift over time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import BaseModel, Field

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ProvenanceError

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: Final = "manifest.json"
REPORT_FILENAME: Final = "report.txt"
DIFF_FILENAME: Final = "uncommitted.diff"
INDEX_FILENAME: Final = "index.jsonl"

# Libraries whose version can move a metric. Absent packages are omitted rather
# than recorded as null, so a manifest never claims to pin what it did not see.
TRACKED_PACKAGES: Final = (
    "xgboost",
    "scikit-learn",
    "shap",
    "numpy",
    "pandas",
    "pyarrow",
)

# DeltxConfig fields that change results. Paths (model_cache_dir, classifier_path)
# are locations rather than levers, and batch_size is unused during extraction, so
# none of them belong in a reproducibility record.
RESULT_AFFECTING_CONFIG: Final = (
    "model_name",
    "device",
    "low_surprisal_threshold",
    "max_sequence_length",
    "surprisal_stride",
    "confidence_threshold",
    "random_seed",
)

_HASH_CHUNK_BYTES: Final = 1 << 20
_GIT_TIMEOUT_SECONDS: Final = 15
_SLUG_PATTERN: Final = re.compile(r"[^a-z0-9]+")
_AI_LABEL: Final = 1


class GitState(BaseModel):
    """The code state a run executed against."""

    commit: str | None = None
    branch: str | None = None
    dirty: bool = False
    dirty_files: list[str] = Field(default_factory=list)
    diff_sha256: str | None = None
    diff_path: str | None = None


class Provenance(BaseModel):
    """Where, when, and with what a run executed."""

    timestamp_utc: str
    duration_seconds: float | None = None
    argv: list[str] = Field(default_factory=list)
    python_version: str
    platform: str
    packages: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    git: GitState = Field(default_factory=GitState)


class DatasetFingerprint(BaseModel):
    """Which data a run consumed, precisely enough to detect substitution."""

    features_path: str
    sha256: str
    rows_available: int
    rows_used: int
    class_balance: dict[str, int] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)
    generator_counts: dict[str, int] = Field(default_factory=dict)


class SplitSizes(BaseModel):
    """Row counts per split."""

    train: int
    validation: int | None = None
    test: int


class Evaluation(BaseModel):
    """One trained model's parameters and scores."""

    split: SplitSizes | None = None
    best_params: dict[str, Any] = Field(default_factory=dict)
    cv_scores: dict[str, Any] = Field(default_factory=dict)
    early_stopped_iteration: int | None = None
    training_time_seconds: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    shap_mean_abs: dict[str, float] | None = None


class LeaveOneModelOut(BaseModel):
    """The generalization stress test against an unseen generator."""

    holdout_model: str
    unseen_samples: int
    unseen_recall: float
    best_params: dict[str, Any] = Field(default_factory=dict)
    cv_scores: dict[str, Any] = Field(default_factory=dict)
    mixed_metrics: dict[str, Any] = Field(default_factory=dict)


class DomainTransfer(BaseModel):
    """A train-on-one-corpus, score-on-a-disjoint-corpus evaluation.

    Distinct from :class:`LeaveOneModelOut`: LOMO withholds one *generator* while
    the corpus stays fixed, so it measures novelty of the model that wrote the
    code. This withholds the whole *corpus*, measuring whether the detector
    survives a change of provenance. Recorded separately because the headline
    metrics of a transfer run describe a different population and must never be
    compared against stratified in-distribution runs.
    """

    train_sources: list[str] = Field(default_factory=list)
    eval_sources: list[str] = Field(default_factory=list)
    train_rows: int
    eval_rows: int
    eval_class_balance: dict[str, int] = Field(default_factory=dict)
    eval_generator_counts: dict[str, int] = Field(default_factory=dict)
    ai_recall: float
    metrics: dict[str, Any] = Field(default_factory=dict)


class ShippedArtifact(BaseModel):
    """The model actually written to disk.

    Its ``best_params`` are recorded separately from the headline evaluation's on
    purpose: ``ship()`` re-runs the hyperparameter search over the full dataset, so
    the artifact routinely differs from the model whose metrics get published.
    """

    path: str
    sha256: str
    trained_on_rows: int
    best_params: dict[str, Any] = Field(default_factory=dict)
    cv_scores: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """A complete, citable record of one training run."""

    run_id: str
    provenance: Provenance
    dataset: DatasetFingerprint
    headline: Evaluation
    lomo: LeaveOneModelOut | None = None
    transfer: DomainTransfer | None = None
    shipped: ShippedArtifact | None = None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def make_run_id(tag: str | None = None, moment: datetime | None = None) -> str:
    """Build a sortable run id: a UTC timestamp, optionally suffixed with a slug.

    Args:
        tag: Free-text label; non-alphanumerics collapse to hyphens.
        moment: Timestamp to use; defaults to now.

    Returns:
        e.g. ``"2026-07-17T18-42-11Z_gemini-lomo"``.
    """
    stamp = (moment or datetime.now(UTC)).strftime("%Y-%m-%dT%H-%M-%SZ")
    if not tag:
        return stamp
    slug = _SLUG_PATTERN.sub("-", tag.lower()).strip("-")
    return f"{stamp}_{slug}" if slug else stamp


def sha256_file(path: Path) -> str:
    """Hash a file's contents, streaming so large parquets stay off the heap.

    Raises:
        ProvenanceError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise ProvenanceError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _run_git(args: Sequence[str], repo_root: Path) -> str | None:
    """Run one git command, returning raw stdout, or None if it failed.

    Output is deliberately *not* stripped: ``git status --porcelain`` encodes state
    in the first two columns, so an unstaged modification reads ``" M path"`` and
    the leading space is data. Use :func:`_git_line` for single-line output.
    """
    try:
        # Fixed argv, no shell. "git" resolves through PATH deliberately: its
        # install location varies across platforms and pinning one would break.
        result = subprocess.run(
            ["git", *args],  # noqa: S603, S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git %s failed: %s", " ".join(args), exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_line(args: Sequence[str], repo_root: Path) -> str | None:
    """Run a git command whose output is a single line, stripped."""
    output = _run_git(args, repo_root)
    return output.strip() if output is not None else None


def _porcelain_paths(status: str) -> list[str]:
    """Extract paths from ``git status --porcelain`` output.

    Each line is ``XY PATH``: two status columns and a space. Renames render as
    ``R  old -> new``; the whole spec is kept, since the point is to record what was
    dirty, not to parse it back.
    """
    return [line[3:] for line in status.splitlines() if line.strip()]


def capture_git_state(repo_root: Path) -> tuple[GitState, str | None]:
    """Snapshot the repository state, including any uncommitted diff.

    Degrades to an empty :class:`GitState` (with a warning) when git is missing or
    ``repo_root`` is not a repository: an untraceable run is still worth recording,
    but it must not silently look traceable.

    Args:
        repo_root: Directory to inspect.

    Returns:
        The state, and the uncommitted diff text (``None`` when clean). The diff is
        returned rather than embedded so it can be written as a sibling file.
    """
    commit = _git_line(["rev-parse", "HEAD"], repo_root)
    if commit is None:
        logger.warning(
            "No git metadata for %s; this run will not be traceable to a commit",
            repo_root,
        )
        return GitState(), None

    dirty_files = _porcelain_paths(_run_git(["status", "--porcelain"], repo_root) or "")
    state = GitState(
        commit=commit,
        branch=_git_line(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        dirty=bool(dirty_files),
        dirty_files=dirty_files,
    )
    if not state.dirty:
        return state, None

    # Tracked modifications only; untracked files are named in dirty_files but
    # cannot appear here, so a tree dirtied purely by new files yields no diff —
    # in which case diff_path must stay None rather than promise a missing file.
    diff = _run_git(["diff", "HEAD"], repo_root) or ""
    if diff:
        state.diff_sha256 = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        state.diff_path = DIFF_FILENAME
    logger.warning(
        "Working tree is dirty (%d file(s)); capturing the diff alongside the run",
        len(dirty_files),
    )
    return state, diff or None


def capture_package_versions(
    packages: Sequence[str] = TRACKED_PACKAGES,
) -> dict[str, str]:
    """Resolve installed versions for the libraries that can move a metric."""
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            logger.debug("Package %r not installed; omitted from provenance", name)
    return versions


def snapshot_config(config: DeltxConfig) -> dict[str, Any]:
    """Extract the config fields that affect results (see RESULT_AFFECTING_CONFIG)."""
    dumped = config.model_dump(mode="json")
    return {key: dumped[key] for key in RESULT_AFFECTING_CONFIG if key in dumped}


def build_provenance(
    config: DeltxConfig,
    repo_root: Path,
    *,
    timestamp_utc: str,
    duration_seconds: float | None = None,
    argv: Sequence[str] | None = None,
) -> tuple[Provenance, str | None]:
    """Assemble the full provenance block.

    Returns:
        The block, and the uncommitted diff text (``None`` when the tree is clean).
    """
    git_state, diff = capture_git_state(repo_root)
    provenance = Provenance(
        timestamp_utc=timestamp_utc,
        duration_seconds=duration_seconds,
        argv=list(argv if argv is not None else sys.argv[1:]),
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages=capture_package_versions(),
        config=snapshot_config(config),
        git=git_state,
    )
    return provenance, diff


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Count a column's values as a JSON-safe dict, or empty if absent."""
    if column not in frame.columns:
        return {}
    counts = frame[column].value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def fingerprint_dataset(
    features_path: Path, available: pd.DataFrame, used: pd.DataFrame
) -> DatasetFingerprint:
    """Fingerprint the feature matrix a run consumed.

    Args:
        features_path: The file read; hashed to pin its exact bytes.
        available: The frame as loaded, before any rebalancing.
        used: The frame actually trained on.

    Returns:
        The fingerprint, including per-source and per-generator composition.
    """
    generators: dict[str, int] = {}
    if {"label", "ai_model"} <= set(used.columns):
        ai_rows = used.loc[used["label"] == _AI_LABEL, "ai_model"].dropna()
        generators = {str(k): int(v) for k, v in ai_rows.value_counts().items()}

    return DatasetFingerprint(
        features_path=str(features_path),
        sha256=sha256_file(features_path),
        rows_available=len(available),
        rows_used=len(used),
        class_balance=_value_counts(used, "label"),
        source_counts=_value_counts(used, "source_dataset"),
        generator_counts=generators,
    )


def index_row(manifest: RunManifest, run_dir: Path) -> dict[str, Any]:
    """Flatten a manifest to one row for the runs index.

    Deliberately flat and shallow: the index exists so a whole experiment history
    loads with ``pd.read_json(path, lines=True)`` and compares in one table.
    """
    metrics = manifest.headline.metrics
    transfer = manifest.transfer
    return {
        "run_id": manifest.run_id,
        "timestamp_utc": manifest.provenance.timestamp_utc,
        "git_commit": manifest.provenance.git.commit,
        "git_dirty": manifest.provenance.git.dirty,
        "features_sha256": manifest.dataset.sha256,
        "rows_used": manifest.dataset.rows_used,
        "holdout_model": manifest.lomo.holdout_model if manifest.lomo else None,
        "headline_accuracy": metrics.get("accuracy"),
        "headline_f1": metrics.get("f1_score"),
        "headline_auroc": metrics.get("auroc"),
        "unseen_recall": manifest.lomo.unseen_recall if manifest.lomo else None,
        # Flags the headline above as measured on a *different* population, so a
        # reader tabling the index cannot silently compare it against the rest.
        "eval_sources": ",".join(transfer.eval_sources) if transfer else None,
        "transfer_ai_recall": transfer.ai_recall if transfer else None,
        "run_dir": str(run_dir),
    }


def write_run(
    runs_root: Path,
    manifest: RunManifest,
    report_text: str,
    diff_text: str | None = None,
) -> Path:
    """Persist a run: manifest, console report, diff, and an index entry.

    Everything is written UTF-8 explicitly — the report holds box-drawing glyphs
    that Windows' default codepage cannot encode.

    Args:
        runs_root: Directory holding all runs (the index lives here).
        manifest: The record to write.
        report_text: Console output, verbatim.
        diff_text: Uncommitted diff, when the tree was dirty.

    Returns:
        The run's own directory.

    Raises:
        ProvenanceError: If any file cannot be written.
    """
    run_dir = runs_root / manifest.run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if diff_text:
            (run_dir / DIFF_FILENAME).write_text(diff_text, encoding="utf-8")
        (run_dir / REPORT_FILENAME).write_text(report_text, encoding="utf-8")
        (run_dir / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        with (runs_root / INDEX_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_row(manifest, run_dir)) + "\n")
    except OSError as exc:
        raise ProvenanceError(f"Cannot write run capture to {run_dir}: {exc}") from exc

    logger.info("Run captured → %s", run_dir)
    return run_dir


def load_index(runs_root: Path) -> pd.DataFrame:
    """Load the runs index as a frame, for comparing experiments.

    Returns:
        One row per captured run, or an empty frame when nothing is captured yet.
    """
    index_path = runs_root / INDEX_FILENAME
    if not index_path.is_file():
        logger.warning("No run index at %s", index_path)
        return pd.DataFrame()
    return pd.read_json(index_path, lines=True)

"""Chronological 15-D dataset orchestration, separate from scoring formulas."""

import csv
import hashlib
import json
import logging
import platform
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Protocol

from deltx.common.constants import PYTHON_GENERATED_DIRS
from deltx.common.exceptions import ExtractionError
from deltx.common.models import CommitDataVector as DatasetRow
from deltx.detection.inference import AIDetectionInference
from deltx.extraction.checkpoints import CheckpointHistory, git, isolated_repository
from deltx.extraction.topology import dependency_pagerank, pagerank_percentiles
from deltx.scoring.config import ScoringConfig
from deltx.scoring.models import Analysis, Severity
from deltx.scoring.squale.engine import score_checkpoint
from deltx.scoring.squale.formulas import bounded_log, density

logger = logging.getLogger(__name__)

METADATA_COLUMNS = (
    "row_index",
    "commit_sha",
    "commit_timestamp",
    "repository",
    "filter_enabled",
    "repository_head",
    "sonar_version",
    "sonar_profile",
    "sonar_scanner_version",
    "sonar_analysis_id",
    "sonar_project_key",
    "scoring_config_version",
    "scoring_config_sha256",
    "scoring_config_json",
    "ai_files_analyzed",
    "ai_evidence",
    "empty_python_snapshot",
    "unmapped_rules",
    "first_parent_sha",
    "history_order",
    "python_version",
)


class CheckpointAnalyzer(Protocol):
    """Replace Docker/HTTP with a fake for offline orchestration tests."""

    def analyze(self, source: Path, project_key: str, revision: str) -> Analysis:
        """Analyze this isolated Python snapshot and return completed results."""
        ...


@dataclass(frozen=True)
class DatasetCheckpoint:
    """A model row and its separate, nonpredictive provenance."""

    row: DatasetRow
    metadata: dict[str, str | int | bool]


def project_key_for(repository: Path) -> str:
    """One stable key for this local repository path, independent of commit."""
    digest = hashlib.sha256(str(repository.resolve()).encode()).hexdigest()[:20]
    return f"deltx-{digest}"


def build_dataset(
    repository: Path,
    inference: AIDetectionInference,
    analyzer: CheckpointAnalyzer,
    config: ScoringConfig,
    *,
    filter_enabled: bool = True,
) -> Iterator[DatasetCheckpoint]:
    """Process ancestors before descendants, comparing each to its first parent.

    AI scores describe changed surviving Python files. No evidence uses the
    detector's own 0.0 convention and is explicitly identified in metadata.
    Deleted files contribute churn and use PageRank zero in the changed-file mean.
    """
    repository = repository.resolve()
    head = git(repository, "rev-parse", "--verify", "HEAD^{commit}").strip()
    config_json = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    project_key = project_key_for(repository)
    with isolated_repository(repository) as isolated:
        commits = isolated.iter_commits(head)
        history = CheckpointHistory(isolated, commits)
        row_index = 0
        for commit in commits:
            changes = history.changes(commit)
            relevant = (
                tuple(c for c in changes if c.meaningful) if filter_enabled else changes
            )
            if filter_enabled and not relevant:
                continue
            sources = {
                path: source
                for path, source in history.sources(commit.commit_hash).items()
                if not PYTHON_GENERATED_DIRS.intersection(path.parts)
            }
            files = {
                c.new_path: c.new_source
                for c in relevant
                if c.new_path is not None and c.new_path in sources
            }
            detection = inference.analyze_commit(
                files, commit.commit_hash, commit.timestamp, commit.author
            )
            ranks = dependency_pagerank(
                sources,
                alpha=config.pagerank_alpha,
                tolerance=config.pagerank_tolerance,
                max_iterations=config.pagerank_max_iterations,
            )
            # PR always describes semantic changes, including in no-filter mode.
            meaningful_paths = [c.path for c in changes if c.meaningful]
            centrality = (
                sum(ranks.get(path, 0.0) for path in meaningful_paths)
                / len(meaningful_paths)
                if meaningful_paths
                else 0.0
            )
            historical_churn = {
                path: bounded_log(value)
                for path, value in history.volatility(
                    commit,
                    list(sources),
                    horizon=config.churn_horizon,
                    decay=config.churn_decay,
                ).items()
            }
            with TemporaryDirectory(prefix="deltx-python-") as directory:
                snapshot = Path(directory)
                for path, source in sources.items():
                    destination = snapshot / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(source, encoding="utf-8")
                analysis = analyzer.analyze(snapshot, project_key, commit.commit_hash)
            scores = score_checkpoint(
                analysis.issues,
                analysis.measures,
                pagerank_percentiles(ranks),
                historical_churn,
                config,
            )
            if scores.unmapped_rules:
                logger.warning(
                    "unmapped Sonar rules: %s", ", ".join(scores.unmapped_rules)
                )
            counts = Counter(issue.severity for issue in analysis.issues)
            measures = analysis.measures
            row = DatasetRow(
                score_maintainability=scores.maintainability,
                score_correctness=scores.correctness,
                score_security=scores.security,
                score_efficiency=scores.efficiency,
                ai_confidence_pct=detection.ai_confidence_pct,
                loc_added=sum(c.added for c in relevant),
                loc_deleted=sum(c.deleted for c in relevant),
                files_modified_count=len(relevant),
                avg_pagerank_centrality=centrality,
                density_blocker_issues=density(
                    counts[Severity.BLOCKER], measures.ncloc
                ),
                density_critical_issues=density(
                    counts[Severity.CRITICAL], measures.ncloc
                ),
                density_major_issues=density(counts[Severity.MAJOR], measures.ncloc),
                density_minor_issues=density(counts[Severity.MINOR], measures.ncloc),
                cognitive_complexity=measures.cognitive_complexity,
                duplication_density=measures.duplicated_lines_density,
            )
            yield DatasetCheckpoint(
                row,
                {
                    "row_index": row_index,
                    "commit_sha": commit.commit_hash,
                    "commit_timestamp": commit.timestamp.isoformat(),
                    "repository": str(repository),
                    "filter_enabled": filter_enabled,
                    "repository_head": head,
                    "sonar_version": analysis.sonar_version,
                    "sonar_profile": analysis.profile,
                    "sonar_scanner_version": analysis.scanner_version,
                    "sonar_analysis_id": analysis.analysis_id,
                    "sonar_project_key": project_key,
                    "scoring_config_version": config.version,
                    "scoring_config_sha256": config_hash,
                    "scoring_config_json": config_json,
                    "ai_files_analyzed": detection.files_analyzed,
                    "ai_evidence": "scored"
                    if detection.files_analyzed
                    else "no-scoreable-files",
                    "empty_python_snapshot": not sources,
                    "unmapped_rules": json.dumps(scores.unmapped_rules),
                    "first_parent_sha": commit.first_parent or "",
                    "history_order": "reverse-topological-all-reachable",
                    "python_version": platform.python_version(),
                },
            )
            row_index += 1


def write_dataset(checkpoints: Iterator[DatasetCheckpoint], output: Path) -> int:
    """Stage both CSVs, rolling back ordinary publication errors as a pair."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.csv")
    if output.name.endswith(".metadata.csv"):
        raise ExtractionError("dataset output must not end in .metadata.csv")
    for target in (output, metadata_path):
        if target.exists() and not target.is_file():
            raise ExtractionError(f"CSV destination is not a regular file: {target}")
    columns = tuple(DatasetRow.model_fields)
    directory = Path(mkdtemp(prefix=".deltx-output-", dir=output.parent))
    recovery_needed = False
    try:
        model_temp = directory / "model.csv"
        metadata_temp = directory / "metadata.csv"
        count = 0
        with (
            model_temp.open("w", newline="", encoding="utf-8") as model_file,
            metadata_temp.open("w", newline="", encoding="utf-8") as metadata_file,
        ):
            writer = csv.DictWriter(model_file, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            metadata_writer = csv.DictWriter(
                metadata_file, fieldnames=METADATA_COLUMNS, lineterminator="\n"
            )
            metadata_writer.writeheader()
            for checkpoint in checkpoints:
                writer.writerow(checkpoint.row.model_dump())
                metadata_writer.writerow(checkpoint.metadata)
                count += 1
        backups: dict[Path, Path | None] = {}
        for index, target in enumerate((output, metadata_path)):
            backup = directory / f"original-{index}"
            if target.exists():
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        published: list[Path] = []
        try:
            for staged, target in (
                (model_temp, output),
                (metadata_temp, metadata_path),
            ):
                staged.replace(target)
                published.append(target)
        except OSError as exc:
            recovery_needed = True
            try:
                for target in published:
                    original = backups[target]
                    if original is None:
                        target.unlink()
                    else:
                        original.replace(target)
            except OSError as rollback_error:
                raise ExtractionError(
                    "CSV publication and rollback failed; "
                    f"backups retained at {directory}"
                ) from rollback_error
            recovery_needed = False
            raise ExtractionError(
                "CSV publication failed; original outputs restored"
            ) from exc
    finally:
        if not recovery_needed:
            shutil.rmtree(directory)
    return count

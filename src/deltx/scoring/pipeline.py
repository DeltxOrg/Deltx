"""Orchestration pipeline: commit → four ISO/IEC 25010 quality scores.

Exposes :func:`score_commit` as the single entry point that Stage 1 calls
per commit to fill the four target columns. Also provides a CLI via the
``deltx-score`` console script.

Usage::

    # From the command line (fixture mode)
    deltx-score --from-fixture fixtures/sample_issues.json --src . --commit HEAD

    # From Python (Stage 1 integration)
    from deltx.scoring.pipeline import score_commit
    vector = score_commit(
        component_key="my-project",
        source_dir=Path("./checkout"),
        repo_path=Path("./checkout"),
        commit="abc123",
        sonar_client=client,
        normalizer=normalizer,
        hyperparams=hp,
    )
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from deltx.scoring.aggregation import system_scores
from deltx.scoring.call_graph import build_call_graph, compute_all_churn, compute_centrality
from deltx.scoring.models import (
    CommitQualityVector,
    DimensionScore,
    Hyperparams,
    IsoDimension,
    SonarIssue,
    SonarMeasures,
)
from deltx.scoring.scoring import Normalizer, accumulate, density, module_score
from deltx.scoring.sonar_client import SonarClient
from deltx.scoring.weighting import weight_all

logger = logging.getLogger(__name__)


def score_commit(
    component_key: str,
    source_dir: Path,
    repo_path: Path,
    commit: str,
    sonar_client: SonarClient | None = None,
    normalizer: Normalizer | None = None,
    hyperparams: Hyperparams | None = None,
    issues: list[SonarIssue] | None = None,
    measures: SonarMeasures | None = None,
) -> CommitQualityVector:
    """Score a single commit, producing the four ISO/IEC 25010 quality scores.

    This is the main entry point for the scoring module. Stage 1 calls this
    once per commit to fill the ``score_*`` columns of the 15-D vector.

    Args:
        component_key: SonarQube project/component key.
        source_dir: Path to the checked-out source tree at this commit.
        repo_path: Path to the Git repository root (for churn computation).
        commit: The commit SHA being scored.
        sonar_client: Live SonarQube client (optional if ``issues`` are provided).
        normalizer: A fitted :class:`Normalizer` (required).
        hyperparams: Tunable hyperparameters (defaults used if None).
        issues: Pre-fetched issues (skips SonarQube API call if provided).
        measures: Pre-fetched measures (skips SonarQube API call if provided).

    Returns:
        A :class:`CommitQualityVector` with the four quality scores.
    """
    hp = hyperparams or Hyperparams()

    # --- 1. Fetch issues + measures ---
    if issues is None:
        if sonar_client is None:
            raise ValueError("Either sonar_client or issues must be provided")
        issues = sonar_client.fetch_issues(component_key)

    if measures is None and sonar_client is not None:
        measures = sonar_client.fetch_measures(component_key)

    # --- 2. Build call graph + centrality ---
    graph = build_call_graph(source_dir)
    centrality_map = compute_centrality(graph)

    # --- 3. Compute churn ---
    file_paths = list({issue.component for issue in issues})
    churn_map = compute_all_churn(file_paths, repo_path, commit)

    # --- 4. Centrality/churn lookup functions ---
    def centrality_fn(file_path: str) -> float:
        return centrality_map.get(file_path, 0.0)

    def churn_fn(file_path: str) -> float:
        return churn_map.get(file_path, 0.0)

    # --- 5. Weight all issues ---
    weighted_issues = weight_all(issues, centrality_fn, churn_fn, hp)

    # --- 6. Per-module (file) scoring ---
    # Group weighted issues by file.
    issues_by_file: dict[str, list] = defaultdict(list)
    for wi in weighted_issues:
        issues_by_file[wi.issue.component].append(wi)

    # Estimate per-file LOC. If measures are available use ncloc as a total;
    # otherwise fall back to counting lines in each source file.
    total_loc = measures.ncloc if measures else 0
    file_count = max(len(issues_by_file), 1)

    # Use a fitted normalizer.
    if normalizer is None:
        normalizer = _make_default_normalizer()

    per_module_scores: dict[IsoDimension, list[float]] = defaultdict(list)

    for file_path, file_issues in issues_by_file.items():
        # Estimate this file's LOC.
        file_loc = _estimate_file_loc(file_path, source_dir, total_loc, file_count)

        scores = module_score(file_issues, file_loc, normalizer)
        for dim, score in scores.to_dict().items():
            per_module_scores[dim].append(score)

    # If no issues at all, all scores are perfect.
    if not issues_by_file:
        return CommitQualityVector(
            score_maintainability=100.0,
            score_correctness=100.0,
            score_security=100.0,
            score_efficiency=100.0,
        )

    # --- 7. Squale aggregation (system level) ---
    sys_scores = system_scores(per_module_scores, lam=hp.lam)

    vector = CommitQualityVector(
        score_maintainability=sys_scores.get(IsoDimension.MAINTAINABILITY, 100.0),
        score_correctness=sys_scores.get(IsoDimension.CORRECTNESS, 100.0),
        score_security=sys_scores.get(IsoDimension.SECURITY, 100.0),
        score_efficiency=sys_scores.get(IsoDimension.EFFICIENCY, 100.0),
    )

    logger.info("Scored commit %s: %s", commit, vector.to_dict())
    return vector


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``deltx-score``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="deltx-score",
        description="Compute ISO/IEC 25010 quality scores for a commit.",
    )
    parser.add_argument(
        "--component", required=False, default="",
        help="SonarQube component key (unused in fixture mode)",
    )
    parser.add_argument(
        "--src", required=True, type=Path,
        help="Path to the checked-out source tree",
    )
    parser.add_argument(
        "--commit", required=True,
        help="Commit SHA to score",
    )
    parser.add_argument(
        "--from-fixture", type=Path, default=None,
        help="Path to a SonarQube issues JSON fixture file",
    )
    parser.add_argument(
        "--measures-fixture", type=Path, default=None,
        help="Path to a SonarQube measures JSON fixture file",
    )
    parser.add_argument(
        "--normalizer", type=Path, default=None,
        help="Path to a normalizer.json file",
    )
    parser.add_argument(
        "--hyperparams", type=Path, default=None,
        help="Path to a hyperparams.json file",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Load hyperparams.
    hp = Hyperparams()
    if args.hyperparams and args.hyperparams.exists():
        hp = Hyperparams.from_json(args.hyperparams)

    # Load normalizer.
    normalizer = None
    if args.normalizer and args.normalizer.exists():
        normalizer = Normalizer()
        normalizer.load(args.normalizer)

    # Load issues.
    issues = None
    measures = None
    if args.from_fixture:
        issues, measures = SonarClient.from_fixture(
            args.from_fixture,
            args.measures_fixture,
        )

    try:
        vector = score_commit(
            component_key=args.component,
            source_dir=args.src.resolve(),
            repo_path=args.src.resolve(),
            commit=args.commit,
            issues=issues,
            measures=measures,
            normalizer=normalizer,
            hyperparams=hp,
        )
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        sys.exit(1)

    print(json.dumps(vector.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_file_loc(
    file_path: str,
    source_dir: Path,
    total_loc: int,
    file_count: int,
) -> int:
    """Estimate lines of code for a file.

    Tries to read the actual file; falls back to an even split of total_loc.
    """
    full_path = source_dir / file_path
    if full_path.exists() and full_path.is_file():
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            # Count non-empty, non-comment lines.
            loc = sum(
                1
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
            return max(loc, 1)
        except OSError:
            pass

    # Fallback: even split.
    if total_loc > 0:
        return max(total_loc // file_count, 1)
    return 100  # Reasonable default.


def _make_default_normalizer() -> Normalizer:
    """Create a normalizer fitted on synthetic default data.

    Used when no pre-fitted normalizer is provided (e.g. first run).
    The synthetic data covers a reasonable range of penalty densities.
    """
    normalizer = Normalizer()
    normalizer.fit({
        IsoDimension.MAINTAINABILITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        IsoDimension.CORRECTNESS: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        IsoDimension.SECURITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        IsoDimension.EFFICIENCY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    })
    logger.info("Using default synthetic normalizer")
    return normalizer

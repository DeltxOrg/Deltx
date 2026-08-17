"""score_history.py — Build the full 15-column training vector for every commit.

For each commit in the git history this script:
  1. Creates a temporary git worktree (non-destructive checkout).
  2. Runs the SonarScanner via Docker Compose to upload the code snapshot.
  3. Waits for SonarQube to finish processing the background task.
  4. Fetches issues + measures via :class:`SonarClient` (existing module).
  5. Computes the call-graph centrality via :func:`build_call_graph` /
     :func:`compute_centrality` (existing module).
  6. Calls :func:`score_commit` for the four Squale target scores (existing).
  7. Derives all remaining columns from the already-fetched data — no new
     API calls and no duplicated calculations.
  8. Appends one row per commit to a deterministic CSV.

Columns produced (one row per commit):
    project_key, commit_sha, commit_short, date, author, message,
    score_maintainability, score_correctness, score_security, score_efficiency,
    loc_added, loc_deleted, files_modified_count,
    avg_pagerank_centrality,
    density_blocker_issues, density_critical_issues,
    density_major_issues, density_minor_issues,
    cognitive_complexity, duplication_density

Missing metrics are written as an empty string (not 0) so that downstream
ML pipelines can detect and handle them correctly.

Usage::

    ./.venv/bin/python scripts/score_history.py \\
        --project-key pyevolve \\
        --repo-path /Users/praneeshsurendran/Documents/Pyevolve \\
        --branch master \\
        --max-commits 10 \\
        --output scores_history.csv

    # Skip the SonarScanner step (issues already uploaded to SonarQube):
    ./.venv/bin/python scripts/score_history.py \\
        --project-key pyevolve \\
        --repo-path /Users/praneeshsurendran/Documents/Pyevolve \\
        --branch master \\
        --skip-scan \\
        --output scores_history.csv
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
import argparse
import tempfile
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Bootstrap: make the installed package importable when running directly.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

# Existing modules — do not re-implement their logic.
from deltx.scoring.call_graph import (  # noqa: E402
    build_call_graph,
    compute_centrality,
)
from deltx.scoring.models import (  # noqa: E402
    IsoDimension,
    SonarIssue,
    SonarMeasures,
)
from deltx.scoring.pipeline import (  # noqa: E402
    _make_default_normalizer,
    score_commit,
)
from deltx.scoring.sonar_client import SonarClient  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger("score_history")

# ---------------------------------------------------------------------------
# CSV schema — one source of truth.
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    # Identity
    "project_key",
    "commit_sha",
    # Predictive targets  (from score_commit / CommitQualityVector)
    "score_maintainability",
    "score_correctness",
    "score_security",
    "score_efficiency",
    # Evolutionary drivers  (from git diff-tree --numstat)
    "loc_added",
    "loc_deleted",
    "files_modified_count",
    # Topological  (from build_call_graph / compute_centrality)
    "avg_pagerank_centrality",
    # Static analysis signals  (derived from SonarIssue list + SonarMeasures)
    "density_blocker_issues",
    "density_critical_issues",
    "density_major_issues",
    "density_minor_issues",
    "cognitive_complexity",
    "duplication_density",
]

# Sentinel for "metric could not be obtained for this commit".
_MISSING = ""


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env(env_file: Path) -> None:
    """Load key=value pairs from .env into os.environ (does not override)."""
    if env_file.exists():
        with open(env_file) as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _get_commits(
    repo_path: Path,
    branch: str,
    max_commits: int | None,
) -> list[dict[str, str]]:
    """Return commits from oldest to newest as dicts with sha/date/author/message."""
    cmd = ["git", "log", branch, "--reverse", "--format=%H|%ci|%ae|%s"]
    if max_commits:
        cmd += [f"-{max_commits}"]
    result = subprocess.run(  # noqa: S603
        cmd, cwd=repo_path, capture_output=True, text=True, check=True,
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "sha":    parts[0].strip(),
                "date":   parts[1].strip(),
                "author": parts[2].strip(),
                "message": parts[3].strip(),
            })
    return commits


def _get_diff_stats(repo_path: Path, sha: str) -> dict[str, Any]:
    """
    Return loc_added, loc_deleted, files_modified_count for a single commit.

    Uses ``git diff-tree --numstat`` which gives per-file added/deleted counts
    without checking out the tree.  This is the same git integration pattern
    used by :func:`deltx.scoring.call_graph.compute_churn`.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", sha],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"loc_added": _MISSING, "loc_deleted": _MISSING, "files_modified_count": _MISSING, "modified_files": []}

    added_total = 0
    deleted_total = 0
    file_count = 0
    modified_files = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            file_count += 1
            a, d, fpath = parts[0], parts[1], parts[2]
            modified_files.append(fpath)
            # Binary files show "-" for counts.
            if a != "-" and d != "-":
                try:
                    added_total += int(a)
                    deleted_total += int(d)
                except ValueError:
                    pass

    return {
        "loc_added": added_total,
        "loc_deleted": deleted_total,
        "files_modified_count": file_count,
        "modified_files": modified_files,
    }


# ---------------------------------------------------------------------------
# SonarScanner + task-wait helpers
# ---------------------------------------------------------------------------

def _run_sonar_scanner(
    worktree_path: Path,
    project_key: str,
    commit_sha: str,
    token: str,
    deltx_root: Path,
) -> str | None:
    """
    Run sonar-scanner via Docker Compose against a worktree snapshot.

    Returns the SonarQube CE task ID on success, None on scanner failure.
    """
    env = os.environ.copy()
    env["SONAR_TOKEN"] = token
    env["SCAN_DIR"] = str(worktree_path)

    cmd = [
        "docker", "compose",
        "--profile", "scanner",
        "run", "--rm",
        "sonar-scanner",
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.scm.revision={commit_sha}",
        f"-Dsonar.projectVersion={commit_sha[:8]}",
        "-Dsonar.scm.disabled=true",
    ]

    print(f"  → Scanning {commit_sha[:8]}...")
    result = subprocess.run(  # noqa: S603
        cmd, cwd=deltx_root, capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  ✗ Scanner failed:\n{result.stderr[-400:]}")
        return None

    for line in result.stdout.splitlines():
        if "api/ce/task?id=" in line:
            return line.split("api/ce/task?id=")[-1].strip()

    return "NO_TASK_ID"  # Success but task ID not extractable.


def _wait_for_sonar_task(
    base_url: str,
    task_id: str,
    token: str,
    timeout: int = 120,
) -> bool:
    """Poll the SonarQube CE task API until the task finishes."""
    if task_id == "NO_TASK_ID":
        time.sleep(5)
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{base_url}/api/ce/task",
                params={"id": task_id},
                auth=(token, ""),
                timeout=10,
            )
            if resp.status_code == 200:
                status = resp.json().get("task", {}).get("status", "")
                if status == "SUCCESS":
                    return True
                if status in ("FAILED", "CANCELED"):
                    print(f"  ✗ Task {task_id} ended with status: {status}")
                    return False
        except requests.RequestException:
            pass
        time.sleep(5)

    print(f"  ✗ Timed out waiting for task {task_id}")
    return False


# ---------------------------------------------------------------------------
# Metric derivation helpers (reuse already-fetched data)
# ---------------------------------------------------------------------------

def _issue_densities(
    issues: list[SonarIssue],
    ncloc: int,
) -> dict[str, Any]:
    """
    Compute per-severity issue densities (issues per 1 000 LOC).

    Reuses the :class:`SonarIssue` list already fetched by :func:`score_commit`
    — no additional API call.  Returns _MISSING if ncloc == 0.
    """
    if ncloc == 0:
        return {
            "density_blocker_issues":   _MISSING,
            "density_critical_issues":  _MISSING,
            "density_major_issues":     _MISSING,
            "density_minor_issues":     _MISSING,
        }

    counts: dict[str, int] = {
        "BLOCKER": 0, "CRITICAL": 0, "MAJOR": 0, "MINOR": 0,
    }
    for issue in issues:
        sev = issue.severity.upper()
        if sev in counts:
            counts[sev] += 1

    kloc = ncloc / 1_000.0
    return {
        "density_blocker_issues":   round(counts["BLOCKER"]  / kloc, 6),
        "density_critical_issues":  round(counts["CRITICAL"] / kloc, 6),
        "density_major_issues":     round(counts["MAJOR"]    / kloc, 6),
        "density_minor_issues":     round(counts["MINOR"]    / kloc, 6),
    }


def _avg_centrality(
    centrality_map: dict[str, float],
    modified_files: list[str],
) -> Any:
    """
    Average PageRank centrality of the files modified in this commit.
    """
    scores = []
    for fpath in modified_files:
        if fpath in centrality_map:
            scores.append(centrality_map[fpath])
    if not scores:
        return _MISSING
    return round(sum(scores) / len(scores), 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Score every commit in a git repository and export the full "
            "15-column training vector to CSV."
        )
    )
    parser.add_argument("--project-key", required=True, help="SonarQube project key")
    parser.add_argument(
        "--repo-path", type=Path,
        default=Path(os.getenv("SCAN_DIR", ".")),
        help="Path to the git repository (defaults to SCAN_DIR env var)",
    )
    parser.add_argument("--branch", default="master", help="Branch to walk (default: master)")
    parser.add_argument(
        "--max-commits", type=int, default=None,
        help="Limit the number of commits (newest first). Default: all.",
    )
    parser.add_argument(
        "--token", default=os.getenv("SONAR_TOKEN"),
        help="SonarQube token (defaults to SONAR_TOKEN in .env)",
    )
    parser.add_argument(
        "--sonar-url", default="http://localhost:9000",
        help="SonarQube base URL (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scores_history.csv"),
        help="Output CSV path (default: scores_history.csv)",
    )
    parser.add_argument(
        "--skip-scan", action="store_true",
        help="Skip running SonarScanner (issues must already be in SonarQube)",
    )
    args = parser.parse_args()

    if not args.token:
        print("Error: SonarQube token required. Pass --token or set SONAR_TOKEN in .env")
        sys.exit(1)

    deltx_root = Path(__file__).parent.parent
    normalizer = _make_default_normalizer()
    client = SonarClient(base_url=args.sonar_url, token=args.token)

    print(f"Fetching commits from '{args.branch}' in {args.repo_path}...")
    commits = _get_commits(args.repo_path, args.branch, args.max_commits)
    print(f"Found {len(commits)} commits.\n")

    # Open CSV for append so the script is resumable.
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with open(args.output, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for idx, commit in enumerate(commits, start=1):
            sha    = commit["sha"]
            date   = commit["date"]
            author = commit["author"]
            msg    = commit["message"]
            short  = sha[:8]

            print(f"[{idx}/{len(commits)}] {short}  {date}  {msg[:60]}")

            # ---- git diff stats (no checkout needed) ----
            diff_stats = _get_diff_stats(args.repo_path, sha)

            # ---- temporary worktree for source-level analysis ----
            worktree_dir = tempfile.mkdtemp(prefix=f"deltx_{short}_")
            worktree_path = Path(worktree_dir)

            try:
                try:
                    subprocess.run(  # noqa: S603
                        ["git", "worktree", "add", "--detach", worktree_dir, sha],
                        cwd=args.repo_path,
                        capture_output=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"  ✗ Worktree failed: {exc.stderr.decode()[:200]}")
                    _write_missing_row(writer, csvfile, args.project_key, sha, diff_stats)
                    continue

                # ---- SonarScanner (optional) ----
                if not args.skip_scan:
                    task_id = _run_sonar_scanner(
                        worktree_path=worktree_path,
                        project_key=args.project_key,
                        commit_sha=sha,
                        token=args.token,
                        deltx_root=deltx_root,
                    )
                    if task_id is None:
                        _write_missing_row(writer, csvfile, args.project_key, sha, diff_stats)
                        continue
                    if not _wait_for_sonar_task(args.sonar_url, task_id, args.token):
                        _write_missing_row(writer, csvfile, args.project_key, sha, diff_stats)
                        continue

                # ---- Fetch issues + measures (existing SonarClient) ----
                try:
                    issues: list[SonarIssue] = client.fetch_issues(args.project_key)
                    measures: SonarMeasures = client.fetch_measures(args.project_key)
                except Exception as exc:
                    print(f"  ✗ SonarClient failed: {exc}")
                    _write_missing_row(writer, csvfile, args.project_key, sha, diff_stats)
                    continue

                # ---- Build call graph + centrality (existing module) ----
                graph = build_call_graph(worktree_path)
                centrality_map = compute_centrality(graph)

                # ---- Squale target scores (existing pipeline) ----
                try:
                    vector = score_commit(
                        component_key=args.project_key,
                        source_dir=worktree_path,
                        repo_path=worktree_path,
                        commit=sha,
                        issues=issues,
                        measures=measures,
                        normalizer=normalizer,
                    )
                except Exception as exc:
                    print(f"  ✗ score_commit failed: {exc}")
                    _write_missing_row(writer, csvfile, args.project_key, sha, diff_stats)
                    continue

                # ---- Derive remaining columns from already-fetched data ----
                densities = _issue_densities(issues, measures.ncloc)
                avg_pr    = _avg_centrality(centrality_map, diff_stats["modified_files"])

                row: dict[str, Any] = {
                    # Identity
                    "project_key":   args.project_key,
                    "commit_sha":    sha,
                    # Predictive targets
                    "score_maintainability": round(vector.score_maintainability, 4),
                    "score_correctness":     round(vector.score_correctness, 4),
                    "score_security":        round(vector.score_security, 4),
                    "score_efficiency":      round(vector.score_efficiency, 4),
                    # Evolutionary drivers
                    "loc_added":             diff_stats["loc_added"],
                    "loc_deleted":           diff_stats["loc_deleted"],
                    "files_modified_count":  diff_stats["files_modified_count"],
                    # Topological
                    "avg_pagerank_centrality": avg_pr,
                    # Static analysis signals (from SonarMeasures — already fetched)
                    "cognitive_complexity":    measures.cognitive_complexity if measures.cognitive_complexity else _MISSING,
                    "duplication_density":     measures.duplicated_lines_density,
                    **densities,
                }

                writer.writerow(row)
                csvfile.flush()

                print(
                    f"  ✓ M={vector.score_maintainability:.1f}  "
                    f"C={vector.score_correctness:.1f}  "
                    f"S={vector.score_security:.1f}  "
                    f"E={vector.score_efficiency:.1f}  "
                    f"LOC+{diff_stats['loc_added']}/-{diff_stats['loc_deleted']}  "
                    f"PR={avg_pr}\n"
                )

            finally:
                # Always clean up the worktree.
                subprocess.run(  # noqa: S603
                    ["git", "worktree", "remove", "--force", worktree_dir],
                    cwd=args.repo_path,
                    capture_output=True,
                )

    print(f"\nDone! Results written to: {args.output.resolve()}")


def _write_missing_row(
    writer: csv.DictWriter,
    csvfile: Any,
    project_key: str,
    sha: str,
    diff_stats: dict[str, Any],
) -> None:
    """Write a row with _MISSING for all metric columns."""
    row: dict[str, Any] = {
        "project_key":   project_key,
        "commit_sha":    sha,
        "score_maintainability": _MISSING,
        "score_correctness":     _MISSING,
        "score_security":        _MISSING,
        "score_efficiency":      _MISSING,
        "loc_added":             diff_stats.get("loc_added", _MISSING),
        "loc_deleted":           diff_stats.get("loc_deleted", _MISSING),
        "files_modified_count":  diff_stats.get("files_modified_count", _MISSING),
        "avg_pagerank_centrality": _MISSING,
        "density_blocker_issues":  _MISSING,
        "density_critical_issues": _MISSING,
        "density_major_issues":    _MISSING,
        "density_minor_issues":    _MISSING,
        "cognitive_complexity":    _MISSING,
        "duplication_density":     _MISSING,
    }
    writer.writerow(row)
    csvfile.flush()
    print("  ⚠ Wrote missing-metric row.\n")


if __name__ == "__main__":
    main()

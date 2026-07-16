"""AST-based call graph construction and PageRank centrality computation.

Builds a best-effort directed call graph from Python source files using the
``ast`` module, then computes per-file PageRank centrality normalized to [0, 1].
Also provides per-file churn (lines changed) from Git history.

**Important assumption**: Python's dynamic dispatch means static analysis
cannot resolve every call. This module provides a structural *proxy* for
centrality — it only needs to be relatively correct (rank-preserving), not
sound. This is documented and accepted because centrality is one multiplicative
term among several in the weighting formula.
"""

from __future__ import annotations

import ast
import logging
import subprocess
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def build_call_graph(source_dir: Path) -> nx.DiGraph:
    """Build a directed call graph from Python source files.

    Nodes are file paths (relative to ``source_dir``). An edge A → B means
    file A imports from or calls into file B.

    Args:
        source_dir: Root directory containing Python source files.

    Returns:
        A ``networkx.DiGraph`` with file-path nodes and call/import edges.
    """
    graph = nx.DiGraph()
    source_dir = source_dir.resolve()

    # Discover all Python files.
    py_files = sorted(source_dir.rglob("*.py"))
    if not py_files:
        logger.warning("No Python files found in %s", source_dir)
        return graph

    # Build a module name → relative path index for import resolution.
    module_index: dict[str, str] = {}
    for py_file in py_files:
        rel = py_file.relative_to(source_dir)
        rel_str = str(rel)

        # Add the file as a graph node.
        graph.add_node(rel_str)

        # Index by dotted module name (e.g. "pkg/sub/mod.py" → "pkg.sub.mod").
        module_name = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        module_index[module_name] = rel_str

        # Also index the package (directory) for __init__.py files.
        if rel.name == "__init__.py":
            pkg_name = str(rel.parent).replace("/", ".").replace("\\", ".")
            if pkg_name and pkg_name != ".":
                module_index[pkg_name] = rel_str

    # Parse each file and extract import/call edges.
    for py_file in py_files:
        rel_str = str(py_file.relative_to(source_dir))
        try:
            source_code = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source_code, filename=str(py_file))
        except SyntaxError:
            logger.warning("Syntax error in %s — skipping for call graph", rel_str)
            continue
        except Exception:  # noqa: BLE001
            logger.warning("Failed to parse %s — skipping for call graph", rel_str)
            continue

        # Extract imports.
        for node in ast.walk(tree):
            targets = _extract_import_targets(node)
            for target in targets:
                resolved = _resolve_module(target, module_index)
                if resolved is not None and resolved != rel_str:
                    graph.add_edge(rel_str, resolved)

    logger.info(
        "Call graph: %d nodes, %d edges from %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        source_dir,
    )
    return graph


def compute_centrality(graph: nx.DiGraph, alpha: float = 0.85) -> dict[str, float]:
    """Compute PageRank centrality for each node, normalized to [0, 1].

    Nodes in an empty or singleton graph all receive centrality 1.0.
    Leaf/unreachable nodes receive the graph minimum (never hard-coded zero).

    Args:
        graph: Directed call graph from :func:`build_call_graph`.
        alpha: PageRank damping factor.

    Returns:
        Mapping from file path to centrality in [0, 1].
    """
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_nodes() == 1:
        node = list(graph.nodes())[0]
        return {node: 1.0}

    # NetworkX PageRank returns values that sum to 1.0.
    raw_pr: dict[str, float] = dict(nx.pagerank(graph, alpha=alpha))

    # Min-max normalize to [0, 1].
    values = list(raw_pr.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        # All nodes have equal centrality.
        return {node: 1.0 for node in raw_pr}

    return {
        node: (val - min_val) / (max_val - min_val)
        for node, val in raw_pr.items()
    }


def compute_churn(
    file_path: str,
    repo_path: Path,
    commit: str | None = None,
    n_commits: int = 50,
) -> float:
    """Compute lines changed for a file over the last N commits.

    Args:
        file_path: File path relative to the repository root.
        repo_path: Path to the Git repository root.
        commit: Optional commit SHA to scope the log (HEAD if None).
        n_commits: Number of recent commits to look back.

    Returns:
        Total lines added + deleted for the file. Returns 0.0 if Git
        is unavailable or the file has no history.
    """
    cmd = ["git", "log", "--format=", "--numstat"]

    if commit is not None:
        cmd.extend([f"{commit}~{n_commits}..{commit}"])
    else:
        cmd.extend([f"-{n_commits}"])

    cmd.append("--")
    cmd.append(file_path)

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Git not available or timed out for churn(%s)", file_path)
        return 0.0

    if result.returncode != 0:
        logger.debug("git log returned %d for %s", result.returncode, file_path)
        return 0.0

    total = 0.0
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            added = parts[0]
            deleted = parts[1]
            # Binary files show "-" for added/deleted.
            if added != "-" and deleted != "-":
                try:
                    total += int(added) + int(deleted)
                except ValueError:
                    pass

    return total


def compute_all_churn(
    file_paths: list[str],
    repo_path: Path,
    commit: str | None = None,
    n_commits: int = 50,
) -> dict[str, float]:
    """Compute churn for multiple files.

    Args:
        file_paths: List of file paths relative to the repo root.
        repo_path: Path to the Git repository root.
        commit: Optional commit SHA.
        n_commits: Lookback window.

    Returns:
        Mapping from file path to total churn.
    """
    return {
        fp: compute_churn(fp, repo_path, commit, n_commits)
        for fp in file_paths
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_import_targets(node: ast.AST) -> list[str]:
    """Extract module names from import statements."""
    targets: list[str] = []

    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module is not None:
            targets.append(node.module)

    return targets


def _resolve_module(
    module_name: str,
    module_index: dict[str, str],
) -> str | None:
    """Attempt to resolve a dotted module name to a file in the project.

    Tries progressively shorter prefixes of the module name to handle
    ``from package.module import name`` patterns.
    """
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_index:
            return module_index[candidate]
    return None

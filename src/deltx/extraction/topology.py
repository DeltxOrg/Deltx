"""Deterministic static Python import graph (not a runtime call graph)."""

import ast
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path

import networkx as nx

from deltx.common.exceptions import ExtractionError


def dependency_pagerank(
    sources: Mapping[Path, str], *, alpha: float, tolerance: float, max_iterations: int
) -> dict[Path, float]:
    """Edges point importer -> imported in-repository module, including tests.

    Resolve root and conventional src layouts, relative imports and package
    initializers. Dynamic imports and historical unparseable files have no
    inferred outgoing edges; all files remain nodes.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(sources))
    modules: dict[str, Path] = {}
    names: dict[Path, str] = {}
    for path in sorted(sources):
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        full = ".".join(parts)
        modules[full] = path
        if parts and parts[0] == "src":
            parts.pop(0)
        canonical = ".".join(parts)
        modules.setdefault(canonical, path)
        names[path] = canonical
    for path, source in sorted(sources.items()):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = names[path].split(".")
                    if path.name != "__init__.py":
                        package = package[:-1]
                    if node.level > len(package):
                        continue
                    prefix = package[: len(package) - node.level + 1]
                    base = ".".join([*prefix, *([base] if base else [])])
                imports.add(base)
                imports.update(f"{base}.{alias.name}" for alias in node.names)
        for name in sorted(imports):
            parts = name.split(".")
            for length in range(1, len(parts) + 1):
                target = modules.get(".".join(parts[:length]))
                if target is not None and target != path:
                    graph.add_edge(path, target)
    if not sources:
        return {}
    try:
        ranks: dict[Path, float] = nx.pagerank(
            graph,
            alpha=alpha,
            tol=tolerance,
            max_iter=max_iterations,
        )
    except nx.PowerIterationFailedConvergence as exc:
        raise ExtractionError(
            "PageRank did not converge; increase pagerank_max_iterations"
        ) from exc
    return ranks


def pagerank_percentiles(ranks: Mapping[Path, float]) -> dict[Path, float]:
    """C'(f)=ECDF(PR(f)); ties share the upper percentile (a singleton is 1)."""
    values = sorted(ranks.values())
    return {
        path: bisect_right(values, rank) / len(values) for path, rank in ranks.items()
    }

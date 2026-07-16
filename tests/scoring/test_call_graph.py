"""Tests for the AST call graph and centrality module (call_graph.py).

Covers: empty package, single file, cyclic imports, syntax-error files,
isolated nodes, and the canonical 3-file toy package where the "router"
must receive the highest PageRank.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deltx.scoring.call_graph import build_call_graph, compute_centrality


class TestBuildCallGraph:
    """Tests for call graph construction."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should produce an empty graph."""
        graph = build_call_graph(tmp_path)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_single_file_no_calls(self, tmp_path: Path) -> None:
        """A single file with no imports should produce one node, no edges."""
        (tmp_path / "solo.py").write_text("x = 1\n")
        graph = build_call_graph(tmp_path)
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

    def test_syntax_error_skipped(self, tmp_path: Path) -> None:
        """Files with syntax errors should be skipped gracefully."""
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("def f(:\n")  # SyntaxError
        graph = build_call_graph(tmp_path)
        # good.py is added as a node; bad.py is skipped during parse.
        # But it was still added as a node (we add nodes before parsing).
        assert "good.py" in graph.nodes
        assert "bad.py" in graph.nodes  # Node exists, but no edges from it.

    def test_three_file_router_centrality(self, tmp_path: Path) -> None:
        """In a 3-file package, the 'router' imported by both others gets highest centrality."""
        # Create a toy package:
        #   router.py  — imported by handler.py and worker.py
        #   handler.py — imports router
        #   worker.py  — imports router
        (tmp_path / "router.py").write_text("def route(): pass\n")
        (tmp_path / "handler.py").write_text("import router\ndef handle(): router.route()\n")
        (tmp_path / "worker.py").write_text("import router\ndef work(): router.route()\n")

        graph = build_call_graph(tmp_path)
        assert graph.number_of_nodes() == 3

        # handler.py → router.py and worker.py → router.py
        assert graph.has_edge("handler.py", "router.py")
        assert graph.has_edge("worker.py", "router.py")

        centrality = compute_centrality(graph)
        assert centrality["router.py"] > centrality["handler.py"]
        assert centrality["router.py"] > centrality["worker.py"]

    def test_cyclic_imports(self, tmp_path: Path) -> None:
        """Cyclic imports should not crash the graph builder."""
        (tmp_path / "a.py").write_text("import b\n")
        (tmp_path / "b.py").write_text("import a\n")

        graph = build_call_graph(tmp_path)
        assert graph.number_of_nodes() == 2
        assert graph.has_edge("a.py", "b.py")
        assert graph.has_edge("b.py", "a.py")

        # Centrality should still work on cycles.
        centrality = compute_centrality(graph)
        assert len(centrality) == 2

    def test_package_structure(self, tmp_path: Path) -> None:
        """Packages with __init__.py should be handled correctly."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("def f(): pass\n")
        (tmp_path / "main.py").write_text("from pkg import mod\n")

        graph = build_call_graph(tmp_path)
        assert "main.py" in graph.nodes


class TestComputeCentrality:
    """Tests for PageRank centrality computation."""

    def test_empty_graph(self) -> None:
        """Empty graph should return empty centrality map."""
        import networkx as nx
        graph = nx.DiGraph()
        assert compute_centrality(graph) == {}

    def test_single_node(self) -> None:
        """Single node should get centrality 1.0."""
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_node("only.py")
        centrality = compute_centrality(graph)
        assert centrality["only.py"] == 1.0

    def test_isolated_node_gets_minimum(self, tmp_path: Path) -> None:
        """An isolated leaf node should get the graph minimum, not zero."""
        (tmp_path / "hub.py").write_text("def api(): pass\n")
        (tmp_path / "spoke1.py").write_text("import hub\n")
        (tmp_path / "spoke2.py").write_text("import hub\n")
        (tmp_path / "isolated.py").write_text("x = 1\n")  # No imports/importers

        graph = build_call_graph(tmp_path)
        centrality = compute_centrality(graph)

        # Isolated node should have the minimum centrality (0.0 after min-max).
        assert centrality["isolated.py"] == 0.0

        # But the hub should have the maximum.
        assert centrality["hub.py"] == 1.0

    def test_all_equal_centrality(self) -> None:
        """All equal nodes should all get centrality 1.0."""
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_node("a.py")
        graph.add_node("b.py")
        # No edges → all equal PageRank.
        centrality = compute_centrality(graph)
        assert centrality["a.py"] == 1.0
        assert centrality["b.py"] == 1.0

    def test_centrality_bounded_0_1(self, tmp_path: Path) -> None:
        """All centrality values should be in [0, 1]."""
        (tmp_path / "a.py").write_text("import b\nimport c\n")
        (tmp_path / "b.py").write_text("import c\n")
        (tmp_path / "c.py").write_text("x = 1\n")

        graph = build_call_graph(tmp_path)
        centrality = compute_centrality(graph)

        for val in centrality.values():
            assert 0.0 <= val <= 1.0

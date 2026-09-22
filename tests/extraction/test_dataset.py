"""Real Git histories, fake Sonar/detector, exact schema and state preservation."""

import csv
import math
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from deltx.common.exceptions import ExtractionError, GitError
from deltx.common.models import CommitDataVector
from deltx.detection.inference import AIDetectionInference
from deltx.extraction.checkpoints import CheckpointHistory, isolated_repository
from deltx.extraction.cli import cli
from deltx.extraction.dataset import (
    DATASET_COLUMNS,
    DatasetCheckpoint,
    build_dataset,
    write_dataset,
)
from deltx.extraction.git_history import GitRepository
from deltx.extraction.semantics import meaningful_change
from deltx.extraction.topology import dependency_pagerank, pagerank_percentiles
from deltx.scoring.config import ScoringConfig
from deltx.scoring.models import (
    Analysis,
    CleanCodeAttribute,
    RuleCatalog,
    SonarMeasures,
    SonarRuleMetadata,
)

from .conftest import GitRepoBuilder

Builder = Callable[[str], GitRepoBuilder]


class FakeAnalyzer:
    def __init__(self) -> None:
        self.revisions: list[str] = []
        self.snapshots: list[list[str]] = []

    def analyze(self, source: Path, project_key: str, revision: str) -> Analysis:
        self.revisions.append(revision)
        self.snapshots.append(
            sorted(
                p.relative_to(source).as_posix()
                for p in source.rglob("*")
                if p.is_file()
            )
        )
        return Analysis(
            (),
            SonarMeasures(
                ncloc=10 if self.snapshots[-1] else 0,
                cognitive_complexity=0,
                duplicated_lines_density=0,
                sqale_index=0,
            ),
            "9.9.8",
            "test-profile",
            "5.0.1",
            "analysis",
            rule_catalog=RuleCatalog(
                {
                    "python:efficient": SonarRuleMetadata(
                        "python:efficient", CleanCodeAttribute.EFFICIENT
                    ),
                }
            ),
        )


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("x=1\n", "# comment\nx=1\n", False),
        ("x=1\n", "x = (1)\n\n", False),
        ('"old docs"\nx=1\n', '"new docs"\nx=1\n', False),
        (
            "def f():\n    'old'\n    return 1\n",
            "def f():\n    'new'\n    return 1\n",
            False,
        ),
        ("class C:\n    'old'\n    pass\n", "class C:\n    'new'\n    pass\n", False),
        (
            "async def f():\n    'old'\n    pass\n",
            "async def f():\n    'new'\n    pass\n",
            False,
        ),
        ("", "# comment\n'only documentation'\n", False),
        ("x=1\n", "x=2\n", True),
        ("x=1\n", "", True),
        ("print x\n", "print x # comment\n", False),
        ("print x\n", "print y\n", True),
        ("if x:\n print y\n", "if x:\n pass\nprint y\n", True),
        ("unterminated = (", "unterminated = [", True),
        ("'assigned'\nx='old'", "'assigned'\nx='new'", True),
    ],
)
def test_semantic_fingerprints(old: str, new: str, expected: bool) -> None:
    assert meaningful_change(old, new) is expected


def test_selection_churn_and_untouched_worktree(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    repo = repo_builder("source")
    repo.write("README.md", "docs")
    repo.commit("docs root")
    repo.write("a.py", "x = 1\n")
    first = repo.commit("one line")
    repo.write("a.py", "# comment\nx = 1\n")
    repo.commit("comment")
    repo.write("a.py", "'docstring'\nx = 1\n")
    repo.commit("docstring")
    repo.write("a.py", "'docstring'\nx=(1)\n\n")
    repo.commit("format")
    repo.run("mv", "a.py", "renamed.py")
    repo.commit("pure rename")
    repo.write("renamed.py", "'docstring'\nx=2\n\n")
    repo.write("README.md", "lots\nof\nnon\nPython\n")
    second = repo.commit("cleanup")
    repo.run("rm", "renamed.py")
    third = repo.commit("delete Python")
    repo.write("local.py", "untracked = True")
    repo.write("README.md", "staged docs")
    repo.run("add", "README.md")
    repo.write("README.md", "unstaged docs")
    status = repo.run("status", "--porcelain=v1")
    index = (repo.root / ".git/index").read_bytes()
    analyzer = FakeAnalyzer()
    rows = list(build_dataset(repo.root, fake_inference, analyzer, ScoringConfig()))
    assert analyzer.revisions == [first, second, third]
    output = repo.root.parent / "filtered.csv"
    write_dataset(iter(rows), output)
    with output.open() as file:
        exported = list(csv.DictReader(file))
    assert [row["commit_hash"] for row in exported] == [first, second, third]
    assert {row["repository"] for row in exported} == {repo.root.name}
    assert [r.row.files_modified_count for r in rows] == [1, 1, 1]
    assert rows[1].row.loc_added == rows[1].row.loc_deleted == 1
    assert rows[2].row.loc_deleted == 3
    assert rows[2].row.ai_confidence_pct == 0
    assert rows[2].row.avg_pagerank_centrality == 0
    assert rows[2].row.score_maintainability == 100
    assert repo.run("status", "--porcelain=v1") == status
    assert (repo.root / ".git/index").read_bytes() == index
    assert (repo.root / "README.md").read_text() == "unstaged docs"
    assert (repo.root / "local.py").exists()


def test_no_filter_and_exact_csv(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    repo = repo_builder("source")
    repo.write("app.py", "x=1\n")
    repo.write("tests/test_app.py", "assert True\n")
    repo.write(".venv/generated.py", "x=1\n")
    repo.commit("first")
    repo.write("README.md", "docs")
    repo.commit("docs")
    analyzer = FakeAnalyzer()
    rows = list(
        build_dataset(
            repo.root, fake_inference, analyzer, ScoringConfig(), filter_enabled=False
        )
    )
    assert len(rows) == 2
    assert (
        rows[1].row.loc_added
        == rows[1].row.loc_deleted
        == rows[1].row.files_modified_count
        == 0
    )
    assert rows[1].row.avg_pagerank_centrality == 0
    assert analyzer.snapshots == [["app.py", "tests/test_app.py"]] * 2
    output = repo.root.parent / "out.csv"
    assert write_dataset(iter(rows), output) == 2
    with output.open() as file:
        data = list(csv.reader(file))
    expected = [
        "score_maintainability",
        "score_correctness",
        "score_security",
        "score_efficiency",
        "ai_confidence_pct",
        "loc_added",
        "loc_deleted",
        "files_modified_count",
        "avg_pagerank_centrality",
        "density_blocker_issues",
        "density_critical_issues",
        "density_major_issues",
        "density_minor_issues",
        "cognitive_complexity",
        "duplication_density",
    ]
    assert expected == list(CommitDataVector.model_fields)
    assert data[0] == ["repository", "commit_hash", *expected] == list(DATASET_COLUMNS)
    assert all(len(row) == 17 for row in data)
    assert all(math.isfinite(float(value)) for row in data[1:] for value in row[2:])
    with output.with_suffix(".metadata.csv").open() as file:
        metadata = list(csv.DictReader(file))
    assert len(metadata) == 2 and metadata[1]["row_index"] == "1"
    assert metadata[1]["ai_evidence"] == "no-scoreable-files"
    for exported, provenance in zip(data[1:], metadata, strict=True):
        assert provenance["repository"] == str(repo.root.resolve())
        assert exported[:2] == [repo.root.name, provenance["commit_sha"]]
        assert len(exported[1]) == 40
    before = output.read_bytes()
    write_dataset(iter(rows), output)
    assert output.read_bytes() == before


def test_combined_export_keeps_repository_groups_and_commit_order(
    repo_builder: Builder, fake_inference: AIDetectionInference
) -> None:
    # Export only the names and quote punctuation correctly in CSV.
    repositories = [
        repo_builder('owner-a/project, "one"'),
        repo_builder('owner-b/project, "two"'),
    ]
    checkpoints: list[DatasetCheckpoint] = []
    identities: list[tuple[str, str]] = []
    for repository in repositories:
        for value in (1, 2):
            repository.write("app.py", f"x={value}\n")
            sha = repository.commit(f"value {value}")
            identities.append((repository.root.name, sha))
        checkpoints.extend(
            build_dataset(
                repository.root, fake_inference, FakeAnalyzer(), ScoringConfig()
            )
        )
    output = repositories[0].root.parent / "combined.csv"
    assert write_dataset(iter(checkpoints), output) == 4
    with output.open() as file:
        records = list(csv.DictReader(file))
    assert [(r["repository"], r["commit_hash"]) for r in records] == identities
    assert len({r["repository"] for r in records}) == 2


@pytest.mark.parametrize("field,value", [("repository", ""), ("commit_sha", "abc123")])
def test_missing_identity_preserves_previous_csv_pair(
    repo_builder: Builder,
    fake_inference: AIDetectionInference,
    field: str,
    value: str,
) -> None:
    repository = repo_builder("source")
    repository.write("app.py", "x=1\n")
    repository.commit("first")
    rows = list(
        build_dataset(repository.root, fake_inference, FakeAnalyzer(), ScoringConfig())
    )
    output = repository.root.parent / "output.csv"
    write_dataset(iter(rows), output)
    metadata = output.with_suffix(".metadata.csv")
    original = (output.read_bytes(), metadata.read_bytes())
    rows[0].metadata[field] = value
    with pytest.raises(ExtractionError, match="identity|full commit hash"):
        write_dataset(iter(rows), output)
    assert (output.read_bytes(), metadata.read_bytes()) == original


def test_empty_export_still_has_identity_and_feature_headers(tmp_path: Path) -> None:
    output = tmp_path / "empty.csv"
    assert write_dataset(iter(()), output) == 0
    with output.open() as file:
        assert list(csv.reader(file)) == [list(DATASET_COLUMNS)]


def test_code_changing_rename_and_history(repo_builder: Builder) -> None:
    repo = repo_builder("r")
    source = "\n".join(f"x{i}={i}" for i in range(20)) + "\n"
    repo.write("old.py", source)
    repo.commit("first")
    repo.run("mv", "old.py", "new.py")
    repo.write("new.py", source.replace("x0=0", "x0=99"))
    repo.commit("rename and change")
    repo.write("new.py", source.replace("x0=0", "x0=1000"))
    repo.commit("future")
    repository = GitRepository(repo.root)
    commits = repository.iter_commits("HEAD")
    history = CheckpointHistory(repository, commits)
    change = history.changes(commits[1])[0]
    assert change.old_path == Path("old.py") and change.new_path == Path("new.py")
    assert change.meaningful and change.added == change.deleted == 1
    assert (
        history.volatility(commits[0], [Path("old.py")], horizon=2, decay=0.5)[
            Path("old.py")
        ]
        == 0
    )
    assert history.volatility(commits[1], [Path("new.py")], horizon=2, decay=0.5)[
        Path("new.py")
    ] == pytest.approx(20)
    assert history.volatility(commits[2], [Path("new.py")], horizon=2, decay=0.5)[
        Path("new.py")
    ] == pytest.approx(2 / 20 + 0.5 * 20)


def test_import_graph_relative_src_tests_and_ties() -> None:
    sources = {
        Path("src/pkg/__init__.py"): "",
        Path("src/pkg/core.py"): "x=1",
        Path("src/pkg/api.py"): "from . import core",
        Path("tests/test_core.py"): "from pkg.core import x",
    }
    ranks = dependency_pagerank(
        sources, alpha=0.85, tolerance=1e-12, max_iterations=1000
    )
    assert sum(ranks.values()) == pytest.approx(1)
    assert ranks[Path("src/pkg/core.py")] > ranks[Path("tests/test_core.py")]
    assert ranks == dependency_pagerank(
        dict(reversed(list(sources.items()))),
        alpha=0.85,
        tolerance=1e-12,
        max_iterations=1000,
    )
    percentiles = pagerank_percentiles({Path("a.py"): 0.5, Path("b.py"): 0.5})
    assert set(percentiles.values()) == {1}
    assert (
        dependency_pagerank({}, alpha=0.85, tolerance=1e-12, max_iterations=1000) == {}
    )


@pytest.mark.parametrize("flag", ["--no-filter", "-no-filter"])
def test_cli_aliases(
    flag: str,
    repo_builder: Builder,
    fake_inference: AIDetectionInference,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repo_builder("r")
    repo.write("README.md", "docs")
    repo.commit("docs")
    monkeypatch.setenv("SONAR_TOKEN", "unit-test-token")
    monkeypatch.setenv("SONAR_HOST_URL", "http://localhost:19876")
    monkeypatch.chdir(repo.root)  # No developer .env is available in CI.
    analyzer = FakeAnalyzer()
    with patch(
        "deltx.extraction.cli.AIDetectionInference.from_config",
        return_value=fake_inference,
    ):
        with patch(
            "deltx.scoring.sonarqube.scanner.SonarCheckpointAnalyzer",
            return_value=analyzer,
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "dataset",
                    str(repo.root),
                    "--output",
                    str(repo.root.parent / "out.csv"),
                    flag,
                ],
            )
    assert result.exit_code == 0, result.output
    assert len(analyzer.revisions) == 1


def test_no_temporal_leakage_from_side_branch(repo_builder: Builder) -> None:
    repo = repo_builder("r")
    repo.write("a.py", "x=1\n")
    repo.commit("root")
    repo.run("checkout", "-b", "side")
    repo.write("a.py", "x=1\n" * 100)
    repo.commit("side churn")
    repo.run("checkout", "main")
    repo.write("b.py", "x=1\n")
    main = repo.commit("main")
    repo.run("merge", "--no-ff", "side", "-m", "merge")
    repository = GitRepository(repo.root)
    commits = repository.iter_commits("HEAD")
    history = CheckpointHistory(repository, commits)
    checkpoint = next(c for c in commits if c.commit_hash == main)
    assert history.volatility(checkpoint, [Path("a.py")], horizon=50, decay=0.9)[
        Path("a.py")
    ] == pytest.approx(1)


def test_literal_paths_symlinks_and_python_boundary_renames(
    repo_builder: Builder,
) -> None:
    repo = repo_builder("r")
    repo.write("a[1].py", "x=1\n")
    repo.write("a1.py", "x=2\n")
    repo.write("tab\tline\nfile.py", "x=3\n")
    (repo.root / "linked.py").symlink_to("/outside/repository.py")
    repo.commit("first")
    repo.run("mv", "a[1].py", "code.txt")
    repo.commit("leaves Python")
    repo.run("mv", "code.txt", "b.py")
    repo.commit("enters Python")
    repository = GitRepository(repo.root)
    commits = repository.iter_commits("HEAD")
    history = CheckpointHistory(repository, commits)
    assert set(history.sources(commits[0].commit_hash)) == {
        Path("a[1].py"),
        Path("a1.py"),
        Path("tab\tline\nfile.py"),
    }
    assert len(history.changes(commits[0])) == 3
    (deletion,) = history.changes(commits[1])
    (addition,) = history.changes(commits[2])
    assert deletion.old_path == Path("a[1].py") and deletion.deleted == 1
    assert deletion.new_path is None and deletion.added == 0
    assert addition.new_path == Path("b.py") and addition.added == 1
    assert addition.old_path is None and addition.deleted == 0


def test_shallow_history_is_rejected(repo_builder: Builder, tmp_path: Path) -> None:
    repo = repo_builder("r")
    repo.write("a.py", "x=1\n")
    repo.commit("first")
    repo.write("a.py", "x=2\n")
    repo.commit("second")
    shallow = tmp_path / "shallow"
    repo.run("clone", "--depth", "1", repo.root.as_uri(), str(shallow))
    with pytest.raises(GitError, match="complete Git history"):
        with isolated_repository(shallow):
            pytest.fail("shallow source was accepted")


def test_churn_uses_git_lines_for_unicode_separators(repo_builder: Builder) -> None:
    repo = repo_builder("r")
    repo.write("a.py", "value = 'one\u2028two\u0085three'")
    repo.commit("one physical line without a terminal newline")
    repo.write("a.py", "value = 'changed'\n")
    repo.commit("change")
    repo.write("a.py", "value = 'future'\n")
    repo.commit("future")
    repository = GitRepository(repo.root)
    commits = repository.iter_commits("HEAD")
    history = CheckpointHistory(repository, commits)
    assert history.changes(commits[0])[0].added == 1
    assert history.volatility(commits[2], [Path("a.py")], horizon=1, decay=1) == {
        Path("a.py"): 2
    }


@pytest.mark.parametrize("existing", [False, True])
def test_csv_pair_is_restored_if_second_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    output = tmp_path / "dataset.csv"
    metadata = output.with_suffix(".metadata.csv")
    if existing:
        output.write_bytes(b"old data\n")
        metadata.write_bytes(b"old metadata\n")
    replace = Path.replace

    def fail_metadata(source: Path, target: Path) -> Path:
        if source.name == "metadata.csv":
            raise OSError("simulated metadata publication failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_metadata)
    with pytest.raises(ExtractionError, match="original outputs restored"):
        write_dataset(iter(()), output)
    if existing:
        assert output.read_bytes() == b"old data\n"
        assert metadata.read_bytes() == b"old metadata\n"
    else:
        assert not output.exists() and not metadata.exists()
    assert not list(tmp_path.glob(".deltx-output-*"))


def test_csv_rollback_failure_keeps_recovery_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dataset.csv"
    metadata = output.with_suffix(".metadata.csv")
    output.write_bytes(b"old data\n")
    metadata.write_bytes(b"old metadata\n")
    replace = Path.replace

    def fail_restore(source: Path, target: Path) -> Path:
        if source.name in {"metadata.csv", "original-0"}:
            raise OSError("simulated publication/rollback failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_restore)
    with pytest.raises(ExtractionError, match="backups retained") as error:
        write_dataset(iter(()), output)
    (backup,) = tmp_path.glob(".deltx-output-*")
    assert str(backup) in str(error.value)
    assert (backup / "original-0").read_bytes() == b"old data\n"
    assert (backup / "original-1").read_bytes() == b"old metadata\n"


def test_invalid_csv_destinations(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="must not end"):
        write_dataset(iter(()), tmp_path / "out.metadata.csv")
    output = tmp_path / "dataset.csv"
    output.mkdir()
    with pytest.raises(ExtractionError, match="not a regular file"):
        write_dataset(iter(()), output)

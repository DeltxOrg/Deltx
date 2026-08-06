"""Tests for Git history traversal."""

from collections.abc import Callable
from pathlib import PurePosixPath

import pytest

from deltx.common.exceptions import GitError
from deltx.extraction.git_history import GitRepository

from .conftest import GitRepoBuilder

Builder = Callable[[str], GitRepoBuilder]


def _changed(repo: GitRepository, commit_index: int) -> list[str]:
    commits = repo.iter_commits("main")
    return [p.as_posix() for p in repo.changed_python_files(commits[commit_index])]


def test_commits_are_oldest_first(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 1\n")
    b.commit("first")
    b.write("b.py", "y = 2\n")
    b.commit("second")
    b.write("c.py", "z = 3\n")
    b.commit("third")

    repo = GitRepository(b.root)
    commits = repo.iter_commits("main")

    assert [c.message for c in commits] == ["first", "second", "third"]
    assert commits[0].parents == ()
    assert len(commits[1].parents) == 1
    assert commits[0].timestamp <= commits[1].timestamp <= commits[2].timestamp


def test_root_commit_treats_all_py_as_added(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("pkg/a.py", "x = 1\n")
    b.write("pkg/b.py", "y = 2\n")
    b.write("notes.txt", "hi\n")
    b.commit("root")

    assert _changed(GitRepository(b.root), 0) == ["pkg/a.py", "pkg/b.py"]


def test_modified_and_added_between_commits(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 1\n")
    b.commit("first")
    b.write("a.py", "x = 2\n")  # modified
    b.write("new.py", "n = 1\n")  # added
    b.write("data.json", "{}\n")  # ignored: not .py
    b.commit("second")

    assert _changed(GitRepository(b.root), 1) == ["a.py", "new.py"]


def test_deleted_file_is_excluded(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 1\n")
    b.write("gone.py", "g = 1\n")
    b.commit("first")
    b.run("rm", "gone.py")
    b.commit("delete gone")

    assert _changed(GitRepository(b.root), 1) == []


def test_pure_rename_is_skipped(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("old.py", "x = 1\n")
    b.commit("first")
    b.run("mv", "old.py", "new.py")
    b.commit("rename only")

    assert _changed(GitRepository(b.root), 1) == []


def test_rename_with_modification_scores_destination(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("old.py", "x = 1\n")
    b.commit("first")
    b.run("mv", "old.py", "new.py")
    b.write("new.py", "x = 1\ny = 2\nz = 3\nw = 4\n")  # substantially changed
    b.commit("rename and modify")

    assert _changed(GitRepository(b.root), 1) == ["new.py"]


def test_merge_diffs_against_first_parent(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("base.py", "base = 1\n")
    b.commit("base")

    b.run("checkout", "-q", "-b", "feature")
    b.write("feature.py", "feat = 1\n")
    b.commit("feature work")

    b.run("checkout", "-q", "main")
    b.write("main_only.py", "m = 1\n")
    b.commit("main work")

    b.run("merge", "-q", "--no-ff", "feature", "-m", "merge feature")

    repo = GitRepository(b.root)
    commits = repo.iter_commits("main")
    merge = commits[-1]
    assert len(merge.parents) == 2
    # Diffed against first parent (main), the merge introduces feature.py.
    changed = [p.as_posix() for p in repo.changed_python_files(merge)]
    assert changed == ["feature.py"]


def test_read_file_decodes_utf8(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 'héllo'\n")
    sha = b.commit("first")

    content = GitRepository(b.root).read_file(sha, PurePosixPath("a.py"))
    assert content == "x = 'héllo'\n"


def test_read_file_falls_back_to_latin1(repo_builder: Builder) -> None:
    b = repo_builder("r")
    # 0xe9 is 'é' in latin-1 but an invalid UTF-8 start byte on its own.
    b.write_bytes("a.py", b"x = 1  # caf\xe9\n")
    sha = b.commit("first")

    content = GitRepository(b.root).read_file(sha, PurePosixPath("a.py"))
    assert content is not None
    assert content.endswith("café\n")


def test_read_file_returns_none_for_binary(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write_bytes("a.py", b"\x00\x01\x02binary\x00")
    sha = b.commit("first")

    assert GitRepository(b.root).read_file(sha, PurePosixPath("a.py")) is None


def test_resolve_branch_explicit_and_missing(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 1\n")
    b.commit("first")
    repo = GitRepository(b.root)

    assert repo.resolve_branch("main") == "main"
    with pytest.raises(GitError):
        repo.resolve_branch("does-not-exist")


def test_missing_blob_raises(repo_builder: Builder) -> None:
    b = repo_builder("r")
    b.write("a.py", "x = 1\n")
    sha = b.commit("first")

    with pytest.raises(GitError):
        GitRepository(b.root).read_file(sha, PurePosixPath("missing.py"))

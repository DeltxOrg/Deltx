"""Isolated local Git history, rename-aware Python diffs and ancestral churn."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from deltx.common.exceptions import GitError
from deltx.common.process import run_process
from deltx.extraction.git_history import CommitMeta, GitRepository, _run_git
from deltx.extraction.semantics import meaningful_change


def _physical_loc(source: str) -> int:
    """Count Git's LF-delimited lines, including an unterminated final line."""
    return source.count("\n") + int(bool(source) and not source.endswith("\n"))


@dataclass(frozen=True)
class PythonChange:
    """One logical change; renames retain both paths and deletion counts."""

    old_path: Path | None
    new_path: Path | None
    old_source: str
    new_source: str
    added: int
    deleted: int

    @property
    def path(self) -> Path:
        """Destination for renames, old path for deletion."""
        path = self.new_path or self.old_path
        if path is None:
            raise GitError("change has neither source nor destination")
        return path

    @property
    def meaningful(self) -> bool:
        """AST/token-based semantic change, independent of commit message/size."""
        return meaningful_change(self.old_source, self.new_source)


def git(path: Path, *args: str) -> str:
    """Run bounded Git plumbing with captured diagnostics."""
    return str(_run_git(path, list(args)))


@contextmanager
def isolated_repository(source: Path) -> Iterator[GitRepository]:
    """A shared-object, detached temporary clone never writes the source .git.

    No hooks, submodules or repository code are executed. A clone (rather than
    worktree registration) also leaves the user's worktree registry untouched.
    """
    with TemporaryDirectory(prefix="deltx-history-") as directory:
        destination = Path(directory) / "repository"
        run_process(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "clone",
                "--shared",
                "--no-checkout",
                "--",
                str(source.resolve()),
                str(destination),
            ],
            error_type=GitError,
        )
        if git(destination, "rev-parse", "--is-shallow-repository").strip() == "true":
            raise GitError(
                "complete Git history is required; unshallow the source repository"
            )
        yield GitRepository(destination)


class CheckpointHistory:
    """Read historical contents and numstat once per commit, with a local cache."""

    def __init__(self, repository: GitRepository, commits: list[CommitMeta]) -> None:
        self.repository = repository
        self.commits = {commit.commit_hash: commit for commit in commits}
        self.cache: dict[str, tuple[PythonChange, ...]] = {}

    def changes(self, commit: CommitMeta) -> tuple[PythonChange, ...]:
        """Use Git rename detection and numstat, including root and deletions."""
        if commit.commit_hash in self.cache:
            return self.cache[commit.commit_hash]
        revisions = (
            [commit.first_parent, commit.commit_hash]
            if commit.first_parent
            else [commit.commit_hash]
        )
        raw = git(
            self.repository.repo_dir,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "--numstat",
            "-z",
            "--find-renames",
            "--diff-algorithm=myers",
            "--no-ext-diff",
            "--no-textconv",
            *revisions,
        )
        fields = iter(raw.split("\0"))
        changes: list[PythonChange] = []
        for field in fields:
            if not field:
                continue
            added_text, deleted_text, name = field.split("\t", 2)
            old_name, new_name = (name, name) if name else (next(fields), next(fields))
            old_path = Path(old_name) if old_name.endswith(".py") else None
            new_path = Path(new_name) if new_name.endswith(".py") else None
            if old_path is None and new_path is None:
                continue
            # Blob existence distinguishes additions from deletions.
            old_source, old_path = self._source(commit.first_parent, old_path)
            new_source, new_path = self._source(commit.commit_hash, new_path)
            if old_path is None and new_path is None:
                continue
            if added_text == "-" or deleted_text == "-":
                raise GitError(f"binary Python diff at {commit.commit_hash}: {name}")
            added, deleted = int(added_text), int(deleted_text)
            # Crossing the Python boundary is an addition/deletion of Python content.
            if old_path is None:
                added, deleted = _physical_loc(new_source), 0
            elif new_path is None:
                added, deleted = 0, _physical_loc(old_source)
            changes.append(
                PythonChange(old_path, new_path, old_source, new_source, added, deleted)
            )
        self.cache[commit.commit_hash] = tuple(changes)
        return tuple(changes)

    def _source(
        self, revision: str | None, path: Path | None
    ) -> tuple[str, Path | None]:
        if revision is None or path is None:
            return "", None
        listing = git(
            self.repository.repo_dir, "ls-tree", "-z", revision, "--", path.as_posix()
        )
        if not listing:
            return "", None
        # Symlinks are not Python source and must never escape the snapshot.
        if not listing.startswith(("100644 ", "100755 ")):
            return "", None
        source = self.repository.read_file(revision, PurePosixPath(path))
        if source is None:
            raise GitError(f"binary Python source at {revision}:{path}")
        return source, path

    def sources(self, revision: str) -> dict[Path, str]:
        """Tracked regular Python files only, with no symlink dereferencing."""
        listing = git(self.repository.repo_dir, "ls-tree", "-rz", revision)
        sources = {}
        for entry in listing.split("\0"):
            if not entry:
                continue
            info, name = entry.split("\t", 1)
            if info.startswith(("100644 ", "100755 ")) and name.endswith(".py"):
                path = Path(name)
                source = self.repository.read_file(revision, PurePosixPath(path))
                if source is None:
                    raise GitError(f"binary Python source at {revision}:{name}")
                sources[path] = source
        return sources

    def volatility(
        self, commit: CommitMeta, paths: list[Path], *, horizon: int, decay: float
    ) -> dict[Path, float]:
        """V(f,t)=sum(k=1..H) delta**(k-1)*(added+deleted)/max(previous_loc,1).

        Follow only first-parent ancestors, counting every historical checkpoint
        (including filtered ones). Carry identity backward through renames. The
        current change never contributes to its own historical volatility.
        """
        identities: dict[Path, Path | None] = {path: path for path in paths}
        for change in self.changes(commit):
            if change.new_path in identities:
                identities[change.path] = change.old_path
        values = dict.fromkeys(paths, 0.0)
        parent = commit.first_parent
        for lag in range(horizon):
            if parent is None:
                break
            ancestor = self.commits[parent]
            changes = {change.path: change for change in self.changes(ancestor)}
            for path, identity in identities.items():
                if identity is not None and identity in changes:
                    change = changes[identity]
                    values[path] += decay**lag * (
                        (change.added + change.deleted)
                        / max(_physical_loc(change.old_source), 1)
                    )
                    identities[path] = change.old_path
            parent = ancestor.first_parent
        return values

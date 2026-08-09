"""Git history traversal for Stage 1 extraction.

Wraps a cloned repository and exposes exactly what the extraction pipeline
needs: the ordered list of commits on a branch (oldest first), the Python files
each commit added or modified, and the decoded contents of those files at that
commit.

The repository is read through Git plumbing (``git log``, ``git diff``,
``git show``) rather than by checking each commit out into the working tree. A
1,000-commit history means 1,000 checkouts of filesystem churn; reading blobs by
``<rev>:<path>`` produces byte-identical content with no working-tree mutation,
which also keeps a ``--resume`` run safe to interrupt at any point.
"""

import logging
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from deltx.common.constants import BLOB_ENCODINGS, PYTHON_SUFFIX
from deltx.common.exceptions import GitError

logger = logging.getLogger(__name__)

#: Field separator for machine-parsed ``git log`` output. Chosen because it
#: cannot appear in a commit hash, ISO date, author name, or subject line.
_FIELD_SEP = "\x1f"

#: Git status codes whose *destination* path carries added or modified content
#: worth scoring. ``A`` added, ``M`` modified, ``T`` type-change (still a real
#: content change). Renames and copies are handled separately by similarity.
_SCORED_STATUSES = frozenset({"A", "M", "T"})


@dataclass(frozen=True)
class CommitMeta:
    """Identity and lineage of one commit, read in a single pass.

    Attributes:
        commit_hash: Full 40-character SHA.
        timestamp: Author timestamp, timezone-aware (UTC-normalised downstream).
        author: Author name.
        message: First line of the commit message (the subject).
        parents: Parent SHAs. Empty for the root commit; more than one for a
            merge, in which case the first is the mainline parent.
    """

    commit_hash: str
    timestamp: datetime
    author: str
    message: str
    parents: tuple[str, ...] = field(default_factory=tuple)

    @property
    def first_parent(self) -> str | None:
        """The mainline parent, or ``None`` for the root commit.

        Merges are diffed against their first parent only, so a merge's row
        reflects what that merge introduced relative to the branch it landed on
        rather than everything from the merged-in side.
        """
        return self.parents[0] if self.parents else None


def _run_git(
    repo_dir: Path, args: list[str], *, binary: bool = False
) -> bytes | str:
    """Run a Git command in ``repo_dir`` and return its stdout.

    Args:
        repo_dir: Working directory for the command.
        args: Git subcommand and arguments, without the leading ``git``.
        binary: Return raw bytes when ``True`` (for blob contents that may not
            be valid text); decoded ``str`` otherwise.

    Returns:
        Standard output as ``bytes`` or ``str`` per ``binary``.

    Raises:
        GitError: If Git exits non-zero or is not installed.
    """
    try:
        # Arguments are literal and never passed through a shell; "git" is
        # resolved from PATH deliberately for cross-platform portability.
        completed = subprocess.run(
            ["git", *args],  # noqa: S603, S607
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        msg = "git executable not found on PATH"
        raise GitError(msg) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        msg = f"git {' '.join(args)!s} failed ({exc.returncode}): {stderr}"
        raise GitError(msg) from exc

    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace")


def clone_repository(repo_url: str, dest: Path) -> "GitRepository":
    """Clone ``repo_url`` into ``dest`` and return a repository handle.

    A full clone is performed so every commit on every branch is reachable; a
    shallow clone would break oldest-first traversal.

    Args:
        repo_url: Git clone URL.
        dest: Empty directory to clone into.

    Returns:
        A handle over the cloned repository.

    Raises:
        GitError: If the clone fails.
    """
    logger.info("cloning %s into %s", repo_url, dest)
    # `git clone` needs to run from a directory that exists; dest itself is
    # created by clone. Run from the parent to avoid a chicken-and-egg cwd.
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Literal args, no shell; "git" resolved from PATH (see _run_git).
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(dest)],  # noqa: S603, S607
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        msg = "git executable not found on PATH"
        raise GitError(msg) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        msg = f"could not clone {repo_url}: {stderr}"
        raise GitError(msg) from exc
    return GitRepository(dest)


class GitRepository:
    """Read-only view over a cloned repository."""

    def __init__(self, repo_dir: Path) -> None:
        """Wrap an existing clone.

        Args:
            repo_dir: Path to the repository working directory.
        """
        self.repo_dir = repo_dir

    def resolve_branch(self, branch: str | None) -> str:
        """Resolve the branch to traverse to a concrete, existing ref.

        Args:
            branch: Requested branch name, or ``None`` for the repository's
                default branch.

        Returns:
            A ref that ``git log`` can resolve — the local branch if it exists,
            otherwise its ``origin/`` remote-tracking counterpart.

        Raises:
            GitError: If the requested branch resolves to nothing.
        """
        if branch is None:
            return self._default_branch()

        for candidate in (branch, f"origin/{branch}"):
            if self._ref_exists(candidate):
                return candidate
        msg = f"branch {branch!r} not found in the cloned repository"
        raise GitError(msg)

    def _default_branch(self) -> str:
        """Return the remote's default branch, e.g. ``origin/main``."""
        try:
            head = _run_git(
                self.repo_dir, ["rev-parse", "--abbrev-ref", "origin/HEAD"]
            )
            resolved = str(head).strip()
            if resolved and resolved != "origin/HEAD":
                return resolved
        except GitError:
            logger.debug("origin/HEAD not set; falling back to HEAD")
        return "HEAD"

    def _ref_exists(self, ref: str) -> bool:
        """Whether ``ref`` resolves to a commit."""
        try:
            _run_git(
                self.repo_dir,
                ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            )
        except GitError:
            return False
        return True

    def iter_commits(self, branch: str) -> list[CommitMeta]:
        """List every commit on ``branch``, oldest first, with metadata.

        Equivalent to ``git rev-list --reverse`` but read in a single pass so
        each commit's timestamp, author, subject, and parents come for free.
        Merges are included; no commit is skipped.

        Args:
            branch: A resolved ref (see :meth:`resolve_branch`).

        Returns:
            Commit metadata in chronological order, index 0 being the oldest.
        """
        fmt = _FIELD_SEP.join(["%H", "%aI", "%an", "%s", "%P"])
        raw = str(
            _run_git(
                self.repo_dir,
                ["log", "--reverse", f"--pretty=format:{fmt}", branch],
            )
        )
        commits: list[CommitMeta] = []
        for line in raw.splitlines():
            if not line:
                continue
            sha, iso, author, subject, parents = line.split(_FIELD_SEP)
            commits.append(
                CommitMeta(
                    commit_hash=sha,
                    timestamp=datetime.fromisoformat(iso),
                    author=author,
                    message=subject,
                    parents=tuple(parents.split()) if parents else (),
                )
            )
        return commits

    def changed_python_files(self, commit: CommitMeta) -> list[PurePosixPath]:
        """Python files ``commit`` added or modified, relative to first parent.

        For the root commit (no parent) every ``.py`` file in the tree is an
        addition. For every other commit the diff is taken against the first
        parent, so merges reflect only what they introduced to the mainline.

        Deletions are excluded — a deleted file has no content to score. A pure
        rename (100% similarity, no content change) is excluded too; a rename
        that also modified content contributes its destination path.

        Args:
            commit: The commit to inspect.

        Returns:
            Destination paths of added/modified ``.py`` files, deduplicated and
            sorted for deterministic ordering.
        """
        parent = commit.first_parent
        if parent is None:
            paths = self._root_python_files(commit.commit_hash)
        else:
            paths = self._diff_python_files(parent, commit.commit_hash)
        return sorted(set(paths))

    def _root_python_files(self, commit_hash: str) -> list[PurePosixPath]:
        """Every ``.py`` file in the tree of the root commit."""
        raw = str(
            _run_git(self.repo_dir, ["ls-tree", "-r", "--name-only", commit_hash])
        )
        return [
            PurePosixPath(line)
            for line in raw.splitlines()
            if line.endswith(PYTHON_SUFFIX)
        ]

    def _diff_python_files(
        self, parent: str, commit_hash: str
    ) -> list[PurePosixPath]:
        """Added/modified ``.py`` files between ``parent`` and ``commit_hash``.

        Parses ``--name-status -z`` so rename and copy records (which carry two
        NUL-separated paths) are read correctly and their destination taken.
        """
        raw = str(
            _run_git(
                self.repo_dir,
                [
                    "diff",
                    "--name-status",
                    "--find-renames",
                    "-z",
                    parent,
                    commit_hash,
                ],
            )
        )
        return list(self._parse_name_status(raw))

    @staticmethod
    def _parse_name_status(raw: str) -> Iterator[PurePosixPath]:
        """Yield destination paths of scoreable ``.py`` changes from ``-z`` output.

        The ``-z`` stream is a flat sequence of NUL-terminated fields. A plain
        change is ``status\\0path``; a rename or copy is ``Rnnn\\0old\\0new``,
        so its status token pulls two path fields instead of one.
        """
        fields = raw.split("\x00")
        i = 0
        while i < len(fields):
            status = fields[i]
            if not status:
                break
            code = status[0]
            if code in ("R", "C"):
                # Rename/copy: status, old path, new path.
                similarity = int(status[1:] or "0")
                new_path = fields[i + 2]
                i += 3
                # A 100%-similar rename changed no content: nothing to score.
                if similarity < 100 and new_path.endswith(PYTHON_SUFFIX):
                    yield PurePosixPath(new_path)
                continue
            path = fields[i + 1]
            i += 2
            if code in _SCORED_STATUSES and path.endswith(PYTHON_SUFFIX):
                yield PurePosixPath(path)

    def read_file(
        self, commit_hash: str, path: PurePosixPath
    ) -> str | None:
        """Read and decode a file's contents at a given commit.

        Args:
            commit_hash: Commit whose tree to read from.
            path: Repository-relative path of the file.

        Returns:
            The decoded text, or ``None`` when the blob looks binary or cannot
            be decoded by any known encoding. A ``None`` return is a skip, not
            an error: binary or undecodable ``.py`` files are rare and excluded
            from scoring with a warning rather than aborting the run.
        """
        blob = _run_git(
            self.repo_dir, ["show", f"{commit_hash}:{path.as_posix()}"], binary=True
        )
        assert isinstance(blob, bytes)  # noqa: S101 - narrow for type checker
        if b"\x00" in blob:
            logger.warning("skipping binary file %s at %s", path, commit_hash[:8])
            return None
        for encoding in BLOB_ENCODINGS:
            try:
                return blob.decode(encoding)
            except UnicodeDecodeError:
                continue
        logger.warning(
            "skipping %s at %s: undecodable by %s",
            path,
            commit_hash[:8],
            ", ".join(BLOB_ENCODINGS),
        )
        return None

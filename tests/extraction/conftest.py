"""Fixtures for extraction tests.

These tests use real Git (fast, offline, deterministic) but never the real
detector. A :class:`FakeDetector` standing in for DroidDetect scores by a marker
string in the source, so the production skip / LOC / aggregation code paths run
unchanged while nothing is downloaded.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from deltx.detection.detector import ScoredSource
from deltx.detection.inference import AIDetectionInference
from deltx.detection.models import ClassDistribution

#: Sources containing this marker are scored as machine-generated.
AI_MARKER = "# generated-by-model"


class FakeDetector:
    """A deterministic stand-in for :class:`DroidDetector`.

    Scores ``P(AI) = 0.9`` when :data:`AI_MARKER` appears in the source and
    ``0.1`` otherwise, so tests can assert direction without real weights.
    """

    def score_source(self, source: str) -> ScoredSource:
        """Return a fixed distribution keyed on the marker's presence."""
        p_ai = 0.9 if AI_MARKER in source else 0.1
        distribution = ClassDistribution.from_probabilities([1.0 - p_ai, p_ai])
        return ScoredSource(
            distribution=distribution,
            token_count=max(1, len(source.split())),
            chunk_count=1,
        )


@pytest.fixture
def fake_inference() -> AIDetectionInference:
    """An inference facade backed by :class:`FakeDetector`."""
    return AIDetectionInference(FakeDetector())  # type: ignore[arg-type]


class GitRepoBuilder:
    """Minimal helper to script a real Git repository in a test."""

    def __init__(self, root: Path) -> None:
        """Initialise an empty repo at ``root`` on branch ``main``."""
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.email", "tester@example.com")
        self._git("config", "user.name", "Tester")
        self._git("config", "commit.gpgsign", "false")

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def write(self, rel_path: str, content: str) -> None:
        """Create or overwrite a file, making parent directories as needed."""
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_bytes(self, rel_path: str, content: bytes) -> None:
        """Write raw bytes, for exercising binary/encoding handling."""
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def commit(self, message: str) -> str:
        """Stage everything and commit, returning the new commit SHA."""
        self._git("add", "-A")
        self._git("commit", "-q", "-m", message)
        return self._git("rev-parse", "HEAD").strip()

    def run(self, *args: str) -> str:
        """Run an arbitrary Git command (e.g. ``mv``, ``checkout``, ``merge``)."""
        return self._git(*args)


@pytest.fixture
def repo_builder(tmp_path: Path) -> Callable[[str], GitRepoBuilder]:
    """Factory returning a :class:`GitRepoBuilder` under a named subdirectory."""

    def _make(name: str = "repo") -> GitRepoBuilder:
        return GitRepoBuilder(tmp_path / name)

    return _make

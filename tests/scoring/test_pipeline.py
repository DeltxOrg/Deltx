"""End-to-end pipeline test (Phase 8).

Tests the full score_commit orchestration from fixture input to
CommitQualityVector output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deltx.scoring.models import CommitQualityVector, Hyperparams, IsoDimension
from deltx.scoring.pipeline import score_commit, _make_default_normalizer
from deltx.scoring.sonar_client import SonarClient


# Path to fixtures.
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "deltx" / "scoring" / "fixtures"


class TestScoreCommit:
    """End-to-end tests for the score_commit pipeline."""

    def test_fixture_end_to_end(self, tmp_path: Path) -> None:
        """Loading from fixture → scoring → four floats in [0, 100]."""
        issues_path = FIXTURES_DIR / "sample_issues.json"
        measures_path = FIXTURES_DIR / "sample_measures.json"

        issues, measures = SonarClient.from_fixture(issues_path, measures_path)

        # Create stub source files matching the fixture components.
        for issue in issues:
            file_path = tmp_path / issue.component
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                file_path.write_text(
                    "# Auto-generated stub for testing\n"
                    "import os\n"
                    "x = 1\n" * 20
                )

        vector = score_commit(
            component_key="test",
            source_dir=tmp_path,
            repo_path=tmp_path,
            commit="test_commit_sha",
            issues=issues,
            measures=measures,
            normalizer=_make_default_normalizer(),
            hyperparams=Hyperparams(),
        )

        # Verify output type and range.
        assert isinstance(vector, CommitQualityVector)
        scores = vector.to_dict()
        assert len(scores) == 4

        for name, val in scores.items():
            assert isinstance(val, float), f"{name} is not float"
            assert 0.0 <= val <= 100.0, f"{name} = {val} out of range"

    def test_empty_issues_all_perfect(self, tmp_path: Path) -> None:
        """Zero issues should produce perfect scores."""
        (tmp_path / "empty.py").write_text("")

        vector = score_commit(
            component_key="test",
            source_dir=tmp_path,
            repo_path=tmp_path,
            commit="abc",
            issues=[],
            normalizer=_make_default_normalizer(),
        )

        assert vector.score_maintainability == 100.0
        assert vector.score_correctness == 100.0
        assert vector.score_security == 100.0
        assert vector.score_efficiency == 100.0

    def test_no_client_no_issues_raises(self, tmp_path: Path) -> None:
        """If neither client nor issues are provided, should raise."""
        with pytest.raises(ValueError, match="Either sonar_client or issues"):
            score_commit(
                component_key="test",
                source_dir=tmp_path,
                repo_path=tmp_path,
                commit="abc",
            )

    def test_output_keys_match_contract(self, tmp_path: Path) -> None:
        """Output dict keys must match the 15-D vector field names."""
        (tmp_path / "stub.py").write_text("x = 1\n")

        vector = score_commit(
            component_key="test",
            source_dir=tmp_path,
            repo_path=tmp_path,
            commit="abc",
            issues=[],
            normalizer=_make_default_normalizer(),
        )

        expected_keys = {
            "score_maintainability",
            "score_correctness",
            "score_security",
            "score_efficiency",
        }
        assert set(vector.to_dict().keys()) == expected_keys

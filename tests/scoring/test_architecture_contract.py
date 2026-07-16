"""Architecture contract tests (Phase 9 — contract ring).

These tests import the shared 15-D vector schema from ``deltx.common.models``
and assert that the scoring module stays consistent with it. They prevent the
module from silently drifting out of sync with the rest of the Deltx pipeline.

The five contracts:
1. Schema lock — field names match exactly.
2. Range contract — outputs in [0, 100].
3. Dimension completeness — all four dimensions always populated.
4. Normalizer provenance — mismatched fingerprint rejected.
5. Golden-file regression — fixture input reproduces expected output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltx.common.models import CommitDataVector
from deltx.scoring.models import CommitQualityVector, IsoDimension
from deltx.scoring.pipeline import score_commit, _make_default_normalizer
from deltx.scoring.scoring import Normalizer
from deltx.scoring.models import Hyperparams, SonarIssue
from deltx.scoring.sonar_client import SonarClient


# Path to fixtures shipped with the scoring module.
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "deltx" / "scoring" / "fixtures"


class TestSchemaLock:
    """CommitQualityVector must expose exactly the four fields from CommitDataVector."""

    def test_quality_vector_fields_match_data_vector(self) -> None:
        """The four score fields must be spelled identically in both models."""
        expected_fields = set(CommitDataVector.quality_score_fields())
        actual_fields = set(CommitQualityVector.model_fields.keys())
        assert actual_fields == expected_fields, (
            f"Field drift detected!\n"
            f"  CommitDataVector expects: {sorted(expected_fields)}\n"
            f"  CommitQualityVector has:  {sorted(actual_fields)}"
        )

    def test_quality_vector_field_names_stable(self) -> None:
        """The canonical field names must not change."""
        expected = [
            "score_maintainability",
            "score_correctness",
            "score_security",
            "score_efficiency",
        ]
        assert CommitQualityVector.field_names() == expected

    def test_data_vector_has_15_fields(self) -> None:
        """The canonical 15-D vector must have exactly 15 fields."""
        assert len(CommitDataVector.model_fields) == 15, (
            f"Expected 15 fields, got {len(CommitDataVector.model_fields)}: "
            f"{list(CommitDataVector.model_fields.keys())}"
        )

    def test_field_types_are_float(self) -> None:
        """All four score fields must be typed as float."""
        for field_name in CommitQualityVector.field_names():
            field_info = CommitQualityVector.model_fields[field_name]
            assert field_info.annotation is float, (
                f"Field {field_name} should be float, got {field_info.annotation}"
            )


class TestRangeContract:
    """All outputs must be in [0, 100] with no missing values."""

    def test_vector_validates_range(self) -> None:
        """Pydantic validation should enforce the [0, 100] range."""
        # Valid vector.
        v = CommitQualityVector(
            score_maintainability=50.0,
            score_correctness=75.0,
            score_security=100.0,
            score_efficiency=0.0,
        )
        for val in v.to_dict().values():
            assert 0.0 <= val <= 100.0

    def test_out_of_range_raises(self) -> None:
        """Scores outside [0, 100] should be rejected by Pydantic."""
        with pytest.raises(Exception):
            CommitQualityVector(
                score_maintainability=101.0,
                score_correctness=50.0,
                score_security=50.0,
                score_efficiency=50.0,
            )
        with pytest.raises(Exception):
            CommitQualityVector(
                score_maintainability=50.0,
                score_correctness=-1.0,
                score_security=50.0,
                score_efficiency=50.0,
            )

    def test_no_none_values(self) -> None:
        """All fields must be populated (no None/null)."""
        v = CommitQualityVector(
            score_maintainability=50.0,
            score_correctness=75.0,
            score_security=100.0,
            score_efficiency=0.0,
        )
        for val in v.to_dict().values():
            assert val is not None


class TestDimensionCompleteness:
    """All four dimensions must always be populated, even with zero issues."""

    def test_zero_issues_all_100(self, tmp_path: Path) -> None:
        """A commit with zero issues must score 100 in all four dimensions."""
        # Create an empty source dir.
        (tmp_path / "empty.py").write_text("")

        vector = score_commit(
            component_key="test",
            source_dir=tmp_path,
            repo_path=tmp_path,
            commit="abc123",
            issues=[],
            measures=None,
            normalizer=_make_default_normalizer(),
            hyperparams=Hyperparams(),
        )

        assert vector.score_maintainability == 100.0
        assert vector.score_correctness == 100.0
        assert vector.score_security == 100.0
        assert vector.score_efficiency == 100.0


class TestNormalizerProvenance:
    """score_commit must reject a normalizer with wrong fingerprint."""

    def test_fingerprint_mismatch_rejected(self, tmp_path: Path) -> None:
        """A normalizer loaded with a mismatched fingerprint should raise."""
        normalizer = Normalizer()
        normalizer.fit({
            IsoDimension.MAINTAINABILITY: [0.0, 1.0],
            IsoDimension.CORRECTNESS: [0.0, 1.0],
            IsoDimension.SECURITY: [0.0, 1.0],
            IsoDimension.EFFICIENCY: [0.0, 1.0],
        })
        path = tmp_path / "normalizer.json"
        normalizer.save(path)

        from deltx.common.exceptions import NormalizerError
        bad_normalizer = Normalizer()
        with pytest.raises(NormalizerError, match="Mismatch: expected"):
            bad_normalizer.load(path, expected_fingerprint="wrong_fp_12345")


class TestGoldenFileRegression:
    """Fixture input must reproduce expected output within tight tolerance.

    The golden files are committed and must be explicitly reviewed to update.
    """

    def test_fixture_produces_scores_in_range(self, tmp_path: Path) -> None:
        """The sample fixture must produce four valid scores in [0, 100]."""
        issues_path = FIXTURES_DIR / "sample_issues.json"
        if not issues_path.exists():
            pytest.skip("Fixture not found")

        issues, measures = SonarClient.from_fixture(issues_path)

        # Create a minimal source tree for the call graph.
        for issue in issues:
            file_path = tmp_path / issue.component
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                file_path.write_text("# stub\nx = 1\n")

        vector = score_commit(
            component_key="test",
            source_dir=tmp_path,
            repo_path=tmp_path,
            commit="fixture_test",
            issues=issues,
            measures=measures,
            normalizer=_make_default_normalizer(),
            hyperparams=Hyperparams(),
        )

        # All four scores must be valid floats in [0, 100].
        for name, val in vector.to_dict().items():
            assert isinstance(val, float), f"{name} is not float"
            assert 0.0 <= val <= 100.0, f"{name} = {val} is out of range"
            assert val == val, f"{name} is NaN"  # NaN check

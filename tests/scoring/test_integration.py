"""Opt-in real Docker test; never downloads or invokes the AI model."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from deltx.scoring.config import ScoringConfig
from deltx.scoring.models import Dimension
from deltx.scoring.sonarqube.config import SonarConfig
from deltx.scoring.sonarqube.scanner import SonarCheckpointAnalyzer
from deltx.scoring.squale.engine import score_checkpoint


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DELTX_RUN_SONAR_INTEGRATION") != "1",
    reason="set DELTX_RUN_SONAR_INTEGRATION=1 and SONAR_TOKEN to run Docker",
)
def test_real_python_then_empty_checkpoint(tmp_path: Path) -> None:
    analyzer = SonarCheckpointAnalyzer(SonarConfig())
    source = tmp_path / "source"
    source.mkdir()
    project = f"deltx-integration-{uuid4().hex}"
    module = source / "module.py"
    module.write_text(
        "import hashlib\n\n"
        "def recurse():\n    return recurse()\n\n"
        "def weak_digest(data):\n    return hashlib.md5(data).hexdigest()\n"
    )
    first = analyzer.analyze(source, project, "a" * 40)
    assert first.measures.ncloc > 0
    assert any(issue.rule == "python:S2190" for issue in first.issues)
    assert any(
        impact.dimension == Dimension.SECURITY
        for issue in first.issues
        for impact in issue.impacts
    )
    assert first.scanner_image_id.startswith("sha256:")
    assert first.scanner_version
    scores = score_checkpoint(first.issues, first.measures, {}, {}, ScoringConfig())
    assert scores.correctness < 100 and scores.efficiency < 100
    assert scores.security < 100
    # A second project must not see the first project's findings.
    separate = tmp_path / "separate"
    separate.mkdir()
    isolated = analyzer.analyze(separate, project + "-empty", "c" * 40)
    assert isolated.issues == ()
    assert isolated.measures.ncloc == 0
    module.unlink()
    empty = analyzer.analyze(source, project, "b" * 40)
    assert empty.measures.ncloc == 0
    assert empty.issues == ()
    assert first.analysis_id != empty.analysis_id
    assert first.scanner_image_id == empty.scanner_image_id

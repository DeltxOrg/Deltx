"""Opt-in real Docker test; never downloads or invokes the AI model."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from deltx.common.exceptions import SonarClientError
from deltx.scoring.config import ScoringConfig
from deltx.scoring.models import CleanCodeAttribute, Dimension
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
        "\ndef has_positive(values):\n    return any([value > 0 for value in values])\n"
    )
    first = analyzer.analyze(source, project, "a" * 40)
    assert first.measures.ncloc > 0
    assert any(issue.rule == "python:S2190" for issue in first.issues)
    assert (
        first.rule_catalog.get("python:S2190").clean_code_attribute
        == CleanCodeAttribute.LOGICAL
    )
    assert any(
        first.rule_catalog.get(issue.rule).clean_code_attribute
        == CleanCodeAttribute.EFFICIENT
        for issue in first.issues
    )
    assert any(
        impact.dimension == Dimension.SECURITY
        for issue in first.issues
        for impact in issue.impacts
    )
    assert first.scanner_image_id.startswith("sha256:")
    assert first.scanner_version
    scores = score_checkpoint(
        first.issues, first.measures, {}, {}, ScoringConfig(), first.rule_catalog
    )
    assert scores.correctness < 100 and scores.efficiency < 100
    assert scores.security < 100
    # A second project must not see the first project's findings.
    separate = tmp_path / "separate"
    separate.mkdir()
    (separate / "module.py").write_text("def increment(value):\n    return value + 1\n")
    isolated = analyzer.analyze(separate, project + "-clean", "c" * 40)
    assert isolated.issues == ()
    assert isolated.measures.ncloc > 0
    assert isolated.rule_catalog is first.rule_catalog
    clean = score_checkpoint(
        isolated.issues,
        isolated.measures,
        {},
        {},
        ScoringConfig(),
        isolated.rule_catalog,
    )
    assert (
        clean.maintainability
        == clean.correctness
        == clean.security
        == clean.efficiency
        == 100
    )
    module.write_text("def increment(value):\n    return value + 1\n")
    cleared = analyzer.analyze(source, project, "b" * 40)
    assert cleared.issues == ()
    assert first.analysis_id != cleared.analysis_id
    assert first.scanner_image_id == cleared.scanner_image_id
    module.unlink()
    # Sonar omits required measures for an empty project; do not invent zeros.
    with pytest.raises(SonarClientError, match="missing Sonar measures"):
        analyzer.analyze(source, project, "d" * 40)
    assert analyzer.client.issues(project) == ()

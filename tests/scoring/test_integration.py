"""Opt-in real Docker test; never downloads or invokes the AI model."""

import os
from pathlib import Path

import pytest

from deltx.scoring.sonarqube.config import SonarConfig
from deltx.scoring.sonarqube.scanner import SonarCheckpointAnalyzer


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DELTX_RUN_SONAR_INTEGRATION") != "1",
    reason="set DELTX_RUN_SONAR_INTEGRATION=1 and SONAR_TOKEN to run Docker",
)
def test_real_python_then_empty_checkpoint(tmp_path: Path) -> None:
    analyzer = SonarCheckpointAnalyzer(SonarConfig())
    source = tmp_path / "source"
    source.mkdir()
    module = source / "module.py"
    module.write_text("def increment(value):\n    return value + 1\n")
    first = analyzer.analyze(source, "deltx-integration", "a" * 40)
    assert first.measures.ncloc > 0
    module.unlink()
    empty = analyzer.analyze(source, "deltx-integration", "b" * 40)
    assert empty.measures.ncloc == 0
    assert empty.issues == ()
    assert first.analysis_id != empty.analysis_id

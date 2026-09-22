"""Docker and task boundaries never require an actual Docker daemon."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from deltx.common.exceptions import (
    ConfigurationError,
    DeltxError,
    SonarClientError,
    SonarConnectionRefusedError,
)
from deltx.common.process import run_process
from deltx.scoring.models import SonarMeasures
from deltx.scoring.sonarqube.client import SonarQubeClient
from deltx.scoring.sonarqube.config import SonarConfig
from deltx.scoring.sonarqube.docker import DockerSonarQubeManager
from deltx.scoring.sonarqube.scanner import DockerSonarScanner, SonarCheckpointAnalyzer


def config() -> SonarConfig:
    return SonarConfig(
        token=SecretStr("unit-test-token"), poll_interval=0.001, startup_timeout=0.01
    )


def test_command_is_python_scoped_and_has_identity(tmp_path: Path) -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    scanner = DockerSonarScanner(settings, manager.client, manager)
    command = scanner.command(tmp_path, tmp_path, "project", "revision")
    assert "-Dsonar.inclusions=**/*.py" in command
    assert "-Dsonar.sources=." in command
    assert "-Dsonar.projectKey=project" in command
    assert "-Dsonar.projectVersion=revision" in command
    assert "-Dsonar.scm.revision=revision" in command
    assert "unit-test-token" not in " ".join(command)
    assert "test" not in next(c for c in command if c.startswith("-Dsonar.exclusions"))
    if sys.platform.startswith("linux"):
        assert manager.scanner_url() == settings.host_url
    else:
        assert "host.docker.internal" in manager.scanner_url()
    manager.started = True
    assert "--network" in scanner.command(tmp_path, tmp_path, "project", "revision")
    assert manager.scanner_url() == "http://sonarqube:9000"


def test_existing_linux_loopback_server_uses_host_network(tmp_path: Path) -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    scanner = DockerSonarScanner(settings, manager.client, manager)
    with patch("sys.platform", "linux"):
        assert manager.scanner_url() == "http://localhost:19000"
        command = scanner.command(tmp_path, tmp_path, "p", "sha")
        assert command[command.index("--network") + 1] == "host"


def test_existing_server_preserved() -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(
        manager.client,
        "status",
        return_value={"status": "UP", "version": "9.9.8.100196"},
    ):
        with patch("deltx.scoring.sonarqube.docker.run_process") as run:
            assert manager.ensure_ready() == "9.9.8.100196"
            run.assert_not_called()
    assert not manager.started


def test_manager_start_and_readiness() -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(
        manager.client,
        "status",
        side_effect=[
            SonarConnectionRefusedError("connection refused"),
            {"status": "UP", "version": "9.9.8"},
        ],
    ):
        with patch("deltx.scoring.sonarqube.docker.run_process") as run:
            assert manager.ensure_ready() == "9.9.8"
            assert run.call_count == 2
    assert manager.started


def test_manager_timeout_and_version_error() -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(manager.client, "status", return_value={"status": "STARTING"}):
        with pytest.raises(SonarClientError, match="failed to become healthy"):
            manager.ensure_ready()
    with patch.object(
        manager.client, "status", return_value={"status": "UP", "version": "2026.1"}
    ):
        with pytest.raises(ConfigurationError, match="differs"):
            manager.ensure_ready()


@pytest.mark.parametrize(
    "reason", ["malformed", "HTTP 401", "unavailable or timed out"]
)
def test_manager_does_not_replace_existing_server(reason: str) -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(manager.client, "status", side_effect=SonarClientError(reason)):
        with patch("deltx.scoring.sonarqube.docker.run_process") as run:
            with pytest.raises(SonarClientError):
                manager.ensure_ready()
            run.assert_not_called()


def test_scan_waits_for_exact_task(tmp_path: Path) -> None:
    settings = config()
    client = SonarQubeClient(settings)
    scanner = DockerSonarScanner(
        settings, client, DockerSonarQubeManager(settings, client)
    )

    def run(args: list[str], **kwargs: object) -> bytes:
        volume = next(a for a in args if a.endswith(":/deltx-work"))
        work = Path(volume.removesuffix(":/deltx-work"))
        assert (
            work / "scanner.properties"
        ).read_text() == "sonar.login=unit-test-token\n"
        (work / "report-task.txt").write_text(
            "ceTaskId=exact-task\nceTaskUrl=http://untrusted\n"
        )
        return b"uploaded"

    with patch("deltx.scoring.sonarqube.scanner.run_process", side_effect=run):
        with patch.object(client, "wait_for_task", return_value="analysis") as wait:
            assert scanner.scan(tmp_path, "p", "sha") == "analysis"
            wait.assert_called_once_with("exact-task")


def test_missing_report_and_token(tmp_path: Path) -> None:
    settings = config()
    client = SonarQubeClient(settings)
    scanner = DockerSonarScanner(
        settings, client, DockerSonarQubeManager(settings, client)
    )
    with patch("deltx.scoring.sonarqube.scanner.run_process"):
        with pytest.raises(SonarClientError, match="report-task"):
            scanner.scan(tmp_path, "p", "sha")
    missing = SonarConfig(token=SecretStr(""))
    with pytest.raises(ConfigurationError, match="SONAR_TOKEN"):
        SonarCheckpointAnalyzer(missing).analyze(tmp_path, "p", "sha")


def test_subprocess_no_shell_timeout_diagnostics_and_redaction() -> None:
    with patch("deltx.common.process.subprocess.run") as run:
        run.return_value.stdout = b"ok"
        assert run_process(["git", "status"], timeout=5) == b"ok"
        assert run.call_args.kwargs["timeout"] == 5
        assert run.call_args.kwargs["capture_output"] is True
        assert "shell" not in run.call_args.kwargs
    for error in [
        FileNotFoundError("missing Docker"),
        subprocess.TimeoutExpired("docker", 5),
        subprocess.CalledProcessError(
            1, "docker", output=b"test-secret", stderr=b"failed"
        ),
    ]:
        with patch("deltx.common.process.subprocess.run", side_effect=error):
            with pytest.raises(DeltxError) as raised:
                run_process(["docker"], secrets=("test-secret",))
            assert "test-secret" not in str(raised.value)


def test_config_and_freshness() -> None:
    for fields in [
        {"host_url": "http://external.example"},
        {"host_url": "http://localhost:invalid"},
        {"host_url": "http://localhost:70000"},
        {"scanner_image": "scanner:latest"},
        {"scanner_image": "registry:5000/scanner"},
        {"scanner_host_url": "file:///tmp/sonar"},
        {"expected_version": ""},
    ]:
        with pytest.raises(ValidationError):
            SonarConfig.model_validate(fields)
    analyzer = SonarCheckpointAnalyzer(config())
    with patch.object(
        analyzer.client, "request", return_value={"analyses": [{"key": "wrong"}]}
    ):
        with pytest.raises(SonarClientError, match="another scan"):
            analyzer._verify_latest("p", "expected")


def test_analyzer_collects_only_after_scan_success(tmp_path: Path) -> None:
    analyzer = SonarCheckpointAnalyzer(config())
    measures = SonarMeasures(
        ncloc=0, cognitive_complexity=0, duplicated_lines_density=0
    )
    events: list[str] = []

    def scan(source: Path, key: str, revision: str) -> str:
        events.append("scan completed")
        return "analysis"

    def issues(project_key: str) -> tuple[()]:
        events.append("issues")
        return ()

    with (
        patch.object(analyzer.manager, "ensure_ready", return_value="9.9.8"),
        patch.object(analyzer.scanner, "scan", side_effect=scan),
        patch.object(
            analyzer.client, "request", return_value={"analyses": [{"key": "analysis"}]}
        ),
        patch.object(analyzer.client, "issues", side_effect=issues),
        patch.object(analyzer.client, "measures", return_value=measures),
        patch.object(analyzer.client, "profile", return_value="[]"),
    ):
        result = analyzer.analyze(tmp_path, "p", "sha")
    assert events == ["scan completed", "issues"]
    assert result.analysis_id == "analysis" and result.sonar_version == "9.9.8"


@pytest.mark.parametrize("drift", ["profile", "version"])
def test_analyzer_rejects_configuration_drift(tmp_path: Path, drift: str) -> None:
    analyzer = SonarCheckpointAnalyzer(config())
    measures = SonarMeasures(
        ncloc=1, cognitive_complexity=0, duplicated_lines_density=0
    )
    versions = ["9.9.8", "9.9.8.2" if drift == "version" else "9.9.8"]
    profiles = [
        '[{"key":"py","rulesUpdatedAt":"before"}]',
        '[{"key":"py","rulesUpdatedAt":"after"}]'
        if drift == "profile"
        else '[{"key":"py","rulesUpdatedAt":"before"}]',
    ]
    with (
        patch.object(analyzer.manager, "ensure_ready", side_effect=versions),
        patch.object(analyzer.scanner, "scan", return_value="analysis"),
        patch.object(
            analyzer.client, "request", return_value={"analyses": [{"key": "analysis"}]}
        ),
        patch.object(analyzer.client, "issues", return_value=()),
        patch.object(analyzer.client, "measures", return_value=measures),
        patch.object(analyzer.client, "profile", side_effect=profiles),
    ):
        analyzer.analyze(tmp_path, "p", "one")
        with pytest.raises(ConfigurationError, match="changed during"):
            analyzer.analyze(tmp_path, "p", "two")

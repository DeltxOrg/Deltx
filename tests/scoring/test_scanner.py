"""Docker and task boundaries never require an actual Docker daemon."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from pydantic import SecretStr, ValidationError

from deltx.common.exceptions import (
    ConfigurationError,
    DeltxError,
    SonarClientError,
    SonarConnectionRefusedError,
)
from deltx.common.process import run_process
from deltx.extraction.cli import cli
from deltx.scoring.models import (
    CleanCodeAttribute,
    RuleCatalog,
    SonarMeasures,
    SonarRuleMetadata,
)
from deltx.scoring.sonarqube.client import SonarQubeClient
from deltx.scoring.sonarqube.config import SonarConfig
from deltx.scoring.sonarqube.docker import DockerSonarQubeManager
from deltx.scoring.sonarqube.scanner import DockerSonarScanner, SonarCheckpointAnalyzer


def config() -> SonarConfig:
    return SonarConfig(
        _env_file=None,
        host_url="http://localhost:19000",
        token=SecretStr("unit-test-token"),
        poll_interval=0.001,
        startup_timeout=0.01,
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
    assert manager.scanner_url().endswith(":19000")


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
        return_value={"status": "UP", "version": "26.9.0.129388"},
    ):
        with patch("deltx.scoring.sonarqube.docker.run_process") as run:
            assert manager.ensure_ready() == "26.9.0.129388"
            run.assert_not_called()


def test_manager_start_and_readiness() -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(
        manager.client,
        "status",
        side_effect=[
            SonarConnectionRefusedError("connection refused"),
            {"status": "UP", "version": "26.9"},
        ],
    ):
        with patch("deltx.scoring.sonarqube.docker.run_process") as run:
            assert manager.ensure_ready() == "26.9"
            assert run.call_count == 2


def test_manager_timeout_and_version_error() -> None:
    settings = config()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(manager.client, "status", return_value={"status": "STARTING"}):
        with pytest.raises(SonarClientError, match="failed to become healthy"):
            manager.ensure_ready()
    with patch.object(
        manager.client, "status", return_value={"status": "UP", "version": "9.9.8"}
    ):
        with pytest.raises(ConfigurationError, match="predates"):
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
    scanner.image_id = "sha256:" + "1" * 64

    def run(args: list[str], **kwargs: object) -> bytes:
        volume = next(a for a in args if a.endswith(":/deltx-work"))
        work = Path(volume.removesuffix(":/deltx-work"))
        assert (
            work / "scanner.properties"
        ).read_text() == "sonar.token=unit-test-token\n"
        assert (work / "scanner.properties").stat().st_mode & 0o777 == 0o600
        (work / "report-task.txt").write_text(
            "ceTaskId=exact-task\nceTaskUrl=http://untrusted\n"
        )
        return b"INFO SonarScanner CLI 8.0.0.6341\nuploaded"

    with patch("deltx.scoring.sonarqube.scanner.run_process", side_effect=run):
        with patch.object(client, "wait_for_task", return_value="analysis") as wait:
            assert scanner.scan(tmp_path, "p", "sha") == "analysis"
            wait.assert_called_once_with("exact-task")
            assert scanner.version == "8.0.0.6341"


def test_missing_report_and_token(tmp_path: Path) -> None:
    settings = config()
    client = SonarQubeClient(settings)
    scanner = DockerSonarScanner(
        settings, client, DockerSonarQubeManager(settings, client)
    )
    scanner.image_id = "sha256:" + "1" * 64
    with patch("deltx.scoring.sonarqube.scanner.run_process", return_value=b""):
        with pytest.raises(SonarClientError, match="report-task"):
            scanner.scan(tmp_path, "p", "sha")
    missing = config().model_copy(update={"token": SecretStr("")})
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
        {"scanner_image": "scanner:bad tag"},
        {"scanner_image": "registry:5000/scanner"},
        {"scanner_host_url": "file:///tmp/sonar"},
        {"expected_version": ""},
    ]:
        with pytest.raises(ValidationError):
            SonarConfig.model_validate({"host_url": "http://localhost:19000", **fields})
    analyzer = SonarCheckpointAnalyzer(config())
    with patch.object(
        analyzer.client, "request", return_value={"analyses": [{"key": "wrong"}]}
    ):
        with pytest.raises(SonarClientError, match="another scan"):
            analyzer._verify_latest("p", "expected")


def test_analyzer_collects_only_after_scan_success(tmp_path: Path) -> None:
    analyzer = SonarCheckpointAnalyzer(config())
    measures = SonarMeasures(
        ncloc=0, cognitive_complexity=0, duplicated_lines_density=0, sqale_index=0
    )
    events: list[str] = []

    def scan(source: Path, key: str, revision: str) -> str:
        events.append("scan completed")
        return "analysis"

    def issues(project_key: str) -> tuple[()]:
        events.append("issues")
        return ()

    with (
        patch.object(analyzer.manager, "ensure_ready", return_value="26.9.0"),
        patch.object(analyzer.scanner, "scan", side_effect=scan),
        patch.object(
            analyzer.client, "request", return_value={"analyses": [{"key": "analysis"}]}
        ),
        patch.object(analyzer.client, "issues", side_effect=issues),
        patch.object(analyzer.client, "measures", return_value=measures),
        patch.object(analyzer.client, "profile", return_value='[{"key":"py"}]'),
        patch.object(
            analyzer.client,
            "rule_catalog",
            return_value=RuleCatalog(
                {
                    "python:efficient": SonarRuleMetadata(
                        "python:efficient", CleanCodeAttribute.EFFICIENT
                    ),
                }
            ),
        ) as rules,
    ):
        result = analyzer.analyze(tmp_path, "p", "sha")
    assert events == ["scan completed", "issues"]
    assert result.analysis_id == "analysis" and result.sonar_version == "26.9.0"
    assert result.rule_catalog is rules.return_value


@pytest.mark.parametrize("drift", ["profile", "version", "collection"])
def test_analyzer_rejects_configuration_drift(tmp_path: Path, drift: str) -> None:
    analyzer = SonarCheckpointAnalyzer(config())
    measures = SonarMeasures(
        ncloc=1, cognitive_complexity=0, duplicated_lines_density=0, sqale_index=0
    )
    versions = ["26.9.0", "26.9.1" if drift == "version" else "26.9.0"]
    profiles = [
        '[{"key":"py","rulesUpdatedAt":"before"}]',
        '[{"key":"py","rulesUpdatedAt":"after"}]'
        if drift == "collection"
        else '[{"key":"py","rulesUpdatedAt":"before"}]',
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
        patch.object(
            analyzer.client,
            "rule_catalog",
            return_value=RuleCatalog(
                {
                    "python:efficient": SonarRuleMetadata(
                        "python:efficient", CleanCodeAttribute.EFFICIENT
                    ),
                }
            ),
        ),
    ):
        if drift != "collection":
            analyzer.analyze(tmp_path, "p", "one")
        with pytest.raises(ConfigurationError, match="changed during"):
            analyzer.analyze(tmp_path, "p", "two")


def test_dotenv_is_the_single_url_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SONAR_HOST_URL", raising=False)
    (tmp_path / ".env").write_text("SONAR_HOST_URL=http://127.0.0.1:19876/sonar\n")
    settings = SonarConfig()
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch("deltx.scoring.sonarqube.docker.run_process") as run:
        manager.compose("up")
        assert run.call_args.kwargs["env"]["DELTX_SONAR_BIND"] == "127.0.0.1:19876"
        assert run.call_args.kwargs["env"]["DELTX_SONAR_CONTEXT"] == "/sonar"
    with patch("sys.platform", "darwin"):
        assert manager.scanner_url() == "http://host.docker.internal:19876/sonar"
    with patch("sys.platform", "linux"):
        assert manager.scanner_url() == settings.host_url
    (tmp_path / ".env").unlink()
    with pytest.raises(ValidationError, match="host_url"):
        SonarConfig()


def test_latest_scanner_is_resolved_once_per_run(tmp_path: Path) -> None:
    settings = config()
    client = SonarQubeClient(settings)
    scanner = DockerSonarScanner(
        settings, client, DockerSonarQubeManager(settings, client)
    )
    image_id = "sha256:" + "a" * 64
    with patch(
        "deltx.scoring.sonarqube.scanner.run_process",
        side_effect=[b"pulled", image_id.encode()],
    ) as run:
        scanner.prepare()
        scanner.prepare()
        assert run.call_count == 2
        assert (
            run.call_args_list[0].args[0][-1] == "sonarsource/sonar-scanner-cli:latest"
        )
    assert image_id in scanner.command(tmp_path, tmp_path, "p", "sha")
    assert settings.scanner_image not in scanner.command(tmp_path, tmp_path, "p", "sha")


def test_optional_version_constraint_and_pending_migration() -> None:
    settings = config().model_copy(update={"expected_version": "26.8"})
    manager = DockerSonarQubeManager(settings, SonarQubeClient(settings))
    with patch.object(
        manager.client, "status", return_value={"status": "UP", "version": "26.9"}
    ):
        with pytest.raises(ConfigurationError, match="differs"):
            manager.ensure_ready()
    with patch.object(
        manager.client, "status", return_value={"status": "DB_MIGRATION_NEEDED"}
    ):
        with pytest.raises(ConfigurationError, match="database migration"):
            manager.ensure_ready()


def test_sonar_lifecycle_cli_reads_dotenv_and_preserves_volumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SONAR_HOST_URL", raising=False)
    (tmp_path / ".env").write_text("SONAR_HOST_URL=http://localhost:19876\n")
    with patch.object(DockerSonarQubeManager, "ensure_ready", return_value="26.9"):
        result = CliRunner().invoke(cli, ["sonar", "up"])
        assert result.exit_code == 0, result.output
        assert "http://localhost:19876" in result.output
    with patch("deltx.scoring.sonarqube.docker.run_process") as run:
        result = CliRunner().invoke(cli, ["sonar", "down"])
        assert result.exit_code == 0, result.output
        assert run.call_args.args[0][-1] == "down"
        assert "-v" not in run.call_args.args[0]

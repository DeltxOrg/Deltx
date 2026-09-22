"""Dockerized Python-only scanner and completed-checkpoint collection."""

import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from deltx.common.constants import PYTHON_GENERATED_DIRS
from deltx.common.exceptions import ConfigurationError, SonarClientError
from deltx.common.process import run_process
from deltx.scoring.models import Analysis
from deltx.scoring.sonarqube.client import SonarQubeClient, as_list, as_object
from deltx.scoring.sonarqube.config import SonarConfig
from deltx.scoring.sonarqube.docker import DockerSonarQubeManager


def _properties_escape(value: str) -> str:
    """Escape Java properties values without injecting another property."""
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("=", "\\=")
        .replace(":", "\\:")
        .replace(" ", "\\ ")
    )


class DockerSonarScanner:
    """Scan a prepared snapshot and wait for its exact Compute Engine task."""

    def __init__(
        self,
        config: SonarConfig,
        client: SonarQubeClient,
        manager: DockerSonarQubeManager,
    ) -> None:
        self.config = config
        self.client = client
        self.manager = manager
        self.image_id = ""
        self.version = ""

    def prepare(self) -> None:
        """Refresh the image once, then use its immutable ID for the whole run."""
        if self.image_id:
            return
        run_process(
            ["docker", "pull", self.config.scanner_image],
            timeout=self.config.scan_timeout,
            error_type=SonarClientError,
        )
        image_id = (
            run_process(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    self.config.scanner_image,
                ],
                error_type=SonarClientError,
            )
            .decode()
            .strip()
        )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise SonarClientError("Docker returned an invalid scanner image ID")
        self.image_id = image_id

    def command(
        self, source: Path, work: Path, project_key: str, revision: str
    ) -> list[str]:
        """Literal argv; authentication lives in a temporary protected settings file."""
        args = ["docker", "run", "--rm"]
        if hasattr(os, "getuid"):
            args.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        if sys.platform.startswith("linux"):
            args.extend(["--network", "host"])
        else:
            args.extend(["--add-host", "host.docker.internal:host-gateway"])
        args.extend(
            [
                "--volume",
                f"{source.resolve()}:/usr/src:ro",
                "--volume",
                f"{work.resolve()}:/deltx-work",
                "--env",
                "SONAR_USER_HOME=/tmp/sonar-cache",
                "--workdir",
                "/usr/src",
                self.image_id or self.config.scanner_image,
                "-Dproject.settings=/deltx-work/scanner.properties",
                f"-Dsonar.host.url={self.manager.scanner_url()}",
                f"-Dsonar.projectKey={project_key}",
                f"-Dsonar.projectVersion={revision}",
                f"-Dsonar.scm.revision={revision}",
                "-Dsonar.scm.disabled=true",
                "-Dsonar.sources=.",
                "-Dsonar.inclusions=**/*.py",
                "-Dsonar.exclusions="
                + ",".join(f"**/{d}/**" for d in sorted(PYTHON_GENERATED_DIRS)),
                "-Dsonar.working.directory=/deltx-work/analysis",
                "-Dsonar.scanner.metadataFilePath=/deltx-work/report-task.txt",
                "-Dsonar.sourceEncoding=UTF-8",
            ]
        )
        return args

    def scan(self, source: Path, project_key: str, revision: str) -> str:
        """Upload once and return the completed analysis ID, never stale results."""
        token = self.config.token.get_secret_value()
        if not token:
            raise ConfigurationError(
                "SONAR_TOKEN missing; create a token with "
                "Execute Analysis and Browse permissions"
            )
        self.prepare()
        with TemporaryDirectory(prefix="deltx-scanner-") as directory:
            work = Path(directory)
            settings = work / "scanner.properties"
            settings.touch(mode=0o600)
            settings.write_text(
                f"sonar.token={_properties_escape(token)}\n", encoding="utf-8"
            )
            output = run_process(
                self.command(source, work, project_key, revision),
                timeout=self.config.scan_timeout,
                secrets=(token,),
                error_type=SonarClientError,
            )
            version = re.search(rb"SonarScanner (?:CLI )?(\d[\w.-]*)", output)
            if version:
                self.version = version[1].decode()
            report = work / "report-task.txt"
            if not report.is_file():
                raise SonarClientError("Sonar scan finished without report-task.txt")
            properties = dict(
                line.split("=", 1)
                for line in report.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            task_id = properties.get("ceTaskId")
            if not task_id:
                raise SonarClientError("Sonar scan report has no ceTaskId")
            # Deliberately ignore ceTaskUrl: use only the configured local server.
            return self.client.wait_for_task(task_id)


class SonarCheckpointAnalyzer:
    """Infrastructure adapter used by dataset orchestration."""

    def __init__(self, config: SonarConfig) -> None:
        self.config = config
        self.client = SonarQubeClient(config)
        self.manager = DockerSonarQubeManager(config, self.client)
        self.scanner = DockerSonarScanner(config, self.client, self.manager)
        self.version: str | None = None
        self.profiles: dict[str, str] = {}

    def _verify_latest(self, project_key: str, analysis_id: str) -> None:
        payload = self.client.request(
            "api/project_analyses/search", {"project": project_key, "ps": "1"}
        )
        analyses = as_list(payload.get("analyses"))
        if not analyses or as_object(analyses[0]).get("key") != analysis_id:
            raise SonarClientError(
                "another scan replaced this checkpoint; use an exclusive project key"
            )

    def analyze(self, source: Path, project_key: str, revision: str) -> Analysis:
        """Complete scan, verify freshness, then read active issues and measures."""
        if not self.config.token.get_secret_value():
            raise ConfigurationError(
                "SONAR_TOKEN missing; start the stack and create an analysis token"
            )
        version = self.manager.ensure_ready()
        if self.version is not None and self.version != version:
            raise ConfigurationError(
                "Sonar server version changed during the dataset run"
            )
        self.version = version
        analysis_id = self.scanner.scan(source, project_key, revision)
        self._verify_latest(project_key, analysis_id)
        issues = self.client.issues(project_key)
        measures = self.client.measures(project_key)
        profile = self.client.profile(project_key, empty=measures.ncloc == 0)
        if profile != "[]":
            previous = self.profiles.setdefault(project_key, profile)
            if previous != profile:
                raise ConfigurationError(
                    "Python quality profile changed during the dataset run; "
                    "restore a fixed profile and restart"
                )
        catalog = self.client.rule_catalog(profile)
        for issue in issues:
            catalog.get(issue.rule)
        if self.client.profile(project_key, empty=measures.ncloc == 0) != profile:
            raise ConfigurationError("Python quality profile changed during collection")
        self._verify_latest(project_key, analysis_id)
        return Analysis(
            issues,
            measures,
            self.version,
            profile,
            self.scanner.version,
            analysis_id,
            self.scanner.image_id,
            rule_catalog=catalog,
        )

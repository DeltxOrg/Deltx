"""Manage only the Deltx stack, preserving pre-existing local servers."""

import sys
import time
from urllib.parse import urlsplit

from deltx.common.exceptions import (
    ConfigurationError,
    SonarClientError,
    SonarConnectionRefusedError,
)
from deltx.common.process import run_process
from deltx.scoring.sonarqube.client import SonarQubeClient, as_text
from deltx.scoring.sonarqube.config import SonarConfig


class DockerSonarQubeManager:
    """Start when necessary, wait for UP, never stop any existing instance."""

    def __init__(self, config: SonarConfig, client: SonarQubeClient) -> None:
        self.config = config
        self.client = client
        self.started = False

    def ensure_ready(self) -> str:
        """Return server version or explain Docker, readiness or version failure."""
        reachable = True
        try:
            status = self.client.status()
        except SonarConnectionRefusedError:
            # Only a refused connection authorizes starting the Deltx stack.
            # A timeout/TLS/HTTP/JSON failure could be an existing external server.
            reachable = False
            status = {}
        if not reachable:
            if self.config.host_url.rstrip("/") not in {
                "http://localhost:19000",
                "http://127.0.0.1:19000",
            }:
                raise ConfigurationError(
                    "start the configured local Sonar server; "
                    "automatic stack uses port 19000"
                )
            run_process(["docker", "info"], error_type=SonarClientError)
            if not self.config.compose_file.is_file():
                raise ConfigurationError(
                    f"Sonar Compose file missing: {self.config.compose_file}; "
                    "set SONAR_COMPOSE_FILE"
                )
            run_process(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(self.config.compose_file),
                    "up",
                    "-d",
                    "sonarqube",
                ],
                timeout=self.config.startup_timeout,
                error_type=SonarClientError,
            )
            self.started = True
        deadline = time.monotonic() + self.config.startup_timeout
        last_error = ""
        while time.monotonic() < deadline:
            if status.get("status") == "UP":
                version = as_text(status.get("version"))
                if not (
                    version == self.config.expected_version
                    or version.startswith(f"{self.config.expected_version}.")
                ):
                    raise ConfigurationError(
                        f"Sonar version {version} differs from "
                        f"configured {self.config.expected_version}"
                    )
                return version
            time.sleep(
                min(self.config.poll_interval, max(0, deadline - time.monotonic()))
            )
            try:
                status = self.client.status()
            except SonarClientError as exc:
                last_error = str(exc)
        raise SonarClientError(
            f"SonarQube failed to become healthy at {self.config.host_url}; "
            "inspect docker compose logs sonarqube and host vm.max_map_count. "
            f"{last_error}"
        )

    def scanner_url(self) -> str:
        """Use Docker DNS for our new stack, host gateway for an existing server."""
        if self.config.scanner_host_url:
            return self.config.scanner_host_url
        if self.started:
            return "http://sonarqube:9000"
        if sys.platform.startswith("linux"):
            # Host networking reaches services bound only to Linux loopback.
            return self.config.host_url
        parsed = urlsplit(self.config.host_url)
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://host.docker.internal{port}{parsed.path}"

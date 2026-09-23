"""Manage only the Deltx stack, preserving pre-existing local servers."""

import os
import sys
import time
from typing import Literal
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

    def compose(self, action: Literal["up", "down"]) -> None:
        """Run the managed stack using the host port/context from SONAR_HOST_URL."""
        parsed = urlsplit(self.config.host_url)
        if parsed.scheme != "http":
            raise ConfigurationError(
                "the managed stack serves HTTP; start your HTTPS server separately"
            )
        if not self.config.compose_file.is_file():
            raise ConfigurationError(
                f"Sonar Compose file missing: {self.config.compose_file}; "
                "set SONAR_COMPOSE_FILE"
            )
        address = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
        env = {
            **os.environ,
            "DELTX_SONAR_BIND": f"{address}:{parsed.port or 80}",
            "DELTX_SONAR_CONTEXT": parsed.path.rstrip("/"),
        }
        args = ["docker", "compose", "-f", str(self.config.compose_file), action]
        if action == "up":
            args.extend(["-d", "sonarqube"])
        run_process(
            args,
            env=env,
            timeout=self.config.startup_timeout,
            error_type=SonarClientError,
        )

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
            run_process(["docker", "info"], error_type=SonarClientError)
            self.compose("up")
        deadline = time.monotonic() + self.config.startup_timeout
        last_error = ""
        while time.monotonic() < deadline:
            if status.get("status") == "UP":
                version = as_text(status.get("version"))
                major = version.split(".", 1)[0]
                if not major.isdecimal() or int(major) < 25:
                    raise ConfigurationError(
                        f"Sonar version {version} predates Community Build 25; "
                        "start the latest stack with 'deltx sonar up'"
                    )
                if self.config.expected_version is not None and not (
                    version == self.config.expected_version
                    or version.startswith(f"{self.config.expected_version}.")
                ):
                    raise ConfigurationError(
                        f"Sonar version {version} differs from "
                        f"configured {self.config.expected_version}"
                    )
                return version
            if status.get("status") in {"DB_MIGRATION_NEEDED", "DB_MIGRATION_RUNNING"}:
                raise ConfigurationError(
                    "Sonar database migration is required or in progress; "
                    "complete the supported server update before scoring"
                )
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
        """Route SONAR_HOST_URL through Docker without changing port or context."""
        if sys.platform.startswith("linux"):
            # Host networking reaches services bound only to Linux loopback.
            return self.config.host_url
        parsed = urlsplit(self.config.host_url)
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://host.docker.internal{port}{parsed.path}"

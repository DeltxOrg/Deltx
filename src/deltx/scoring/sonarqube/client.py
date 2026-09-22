"""Validated, paginated SonarQube 9.9 Web API client."""

import base64
import json
import time
from http.client import HTTPException, HTTPMessage
from pathlib import Path
from typing import IO, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import ValidationError

from deltx.common.exceptions import (
    ConfigurationError,
    SonarClientError,
    SonarConnectionRefusedError,
)
from deltx.scoring.models import IssueType, Severity, SonarIssue, SonarMeasures
from deltx.scoring.sonarqube.config import SonarConfig


class _RejectRedirects(HTTPRedirectHandler):
    """Never forward a local server's authentication through a redirect."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> NoReturn:
        raise SonarClientError("unexpected Sonar API redirect; check SONAR_HOST_URL")


def as_object(value: object) -> dict[str, object]:
    """Narrow untrusted JSON immediately at the HTTP boundary."""
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise SonarClientError("malformed Sonar response: expected JSON object")
    return {str(k): v for k, v in value.items()}


def as_list(value: object) -> list[object]:
    """Validate a JSON array before iteration."""
    if not isinstance(value, list):
        raise SonarClientError("malformed Sonar response: expected JSON array")
    return list(value)


def as_text(value: object) -> str:
    """Require a nonempty string for API identity/status fields."""
    if not isinstance(value, str) or not value:
        raise SonarClientError("malformed Sonar response: expected nonempty string")
    return value


class SonarQubeClient:
    """Current issues and measures, retrieved only after CE success by the runner."""

    def __init__(self, config: SonarConfig) -> None:
        self.config = config

    def request(
        self, endpoint: str, params: dict[str, str] | None = None
    ) -> dict[str, object]:
        """GET a local API endpoint, with timeout and actionable errors."""
        url = f"{self.config.host_url.rstrip('/')}/{endpoint}?{urlencode(params or {})}"
        token = self.config.token.get_secret_value()
        headers = {}
        if token:
            encoded = base64.b64encode(f"{token}:".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        request = Request(url, headers=headers)  # noqa: S310 - validated local HTTP(S)
        try:
            # API redirects are unexpected. Reject them rather than forwarding
            # authentication to a different origin or silently using another server.
            opener = build_opener(ProxyHandler({}), _RejectRedirects())
            with opener.open(request, timeout=self.config.request_timeout) as response:
                payload: object = json.load(response)
        except HTTPError as exc:
            raise SonarClientError(
                f"Sonar API {endpoint}: HTTP {exc.code}; "
                "check SONAR_TOKEN and project permissions"
            ) from exc
        except (URLError, OSError, HTTPException) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            if isinstance(reason, ConnectionRefusedError):
                raise SonarConnectionRefusedError(
                    f"no Sonar server listening at {self.config.host_url}"
                ) from exc
            raise SonarClientError(
                f"Sonar API {endpoint} unavailable or timed out: {exc}"
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise SonarClientError(f"malformed Sonar response from {endpoint}") from exc
        return as_object(payload)

    def status(self) -> dict[str, object]:
        """Read status/version without requiring authentication."""
        return self.request("api/system/status")

    def issues(self, project_key: str) -> tuple[SonarIssue, ...]:
        """Retrieve unresolved issues, partitioning by file above Sonar's 10k cap."""
        params = {"componentKeys": project_key, "resolved": "false", "s": "FILE_LINE"}
        first = self.request("api/issues/search", {**params, "p": "1", "ps": "1"})
        total = self._total(first)
        if total <= 10000:
            raw = self._pages("api/issues/search", params, "issues")
        else:
            components = self._pages(
                "api/components/tree",
                {"component": project_key, "qualifiers": "FIL"},
                "components",
            )
            raw = []
            for component in components:
                key = as_text(as_object(component).get("key"))
                raw.extend(
                    self._pages(
                        "api/issues/search", {**params, "componentKeys": key}, "issues"
                    )
                )
            # Project-level issues or concurrent analysis must not silently disappear.
            if len(raw) != total:
                raise SonarClientError(
                    "issue partition totals differ; isolate this project's scans"
                )
        parsed = tuple(self._issue(as_object(item), project_key) for item in raw)
        if len({issue.key for issue in parsed}) != len(parsed):
            raise SonarClientError(
                "duplicate issues across pages; concurrent scan detected"
            )
        return parsed

    @staticmethod
    def _total(payload: dict[str, object]) -> int:
        value = as_object(payload.get("paging")).get("total")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SonarClientError("malformed Sonar pagination total")
        return value

    def _pages(self, endpoint: str, params: dict[str, str], field: str) -> list[object]:
        result: list[object] = []
        page = 1
        expected: int | None = None
        while True:
            payload = self.request(
                endpoint, {**params, "p": str(page), "ps": str(self.config.page_size)}
            )
            total = self._total(payload)
            if expected is not None and total != expected:
                raise SonarClientError("Sonar pagination changed during collection")
            expected = total
            if field == "issues" and total > 10000:
                raise SonarClientError(
                    "more than 10,000 issues in one file; "
                    "API cannot return complete data"
                )
            items = as_list(payload.get(field))
            result.extend(items)
            if len(result) == total:
                return result
            if not items or len(result) > total:
                raise SonarClientError(
                    "malformed Sonar pagination: incomplete/inconsistent page"
                )
            page += 1

    @staticmethod
    def _issue(raw: dict[str, object], project_key: str) -> SonarIssue:
        if "severity" not in raw:
            raise ConfigurationError(
                "Sonar issue has no legacy severity; configure a profile/version "
                "exposing INFO/MINOR/MAJOR/CRITICAL/BLOCKER"
            )
        severity_text = as_text(raw.get("severity"))
        try:
            severity = Severity(severity_text)
        except ValueError as exc:
            raise ConfigurationError(
                f"unsupported Sonar severity {severity_text!r}; "
                "require INFO/MINOR/MAJOR/CRITICAL/BLOCKER"
            ) from exc
        type_text = as_text(raw.get("type"))
        issue_type = (
            IssueType(type_text) if type_text in IssueType else IssueType.UNKNOWN
        )
        component = as_text(raw.get("component"))
        if component != project_key and not component.startswith(f"{project_key}:"):
            raise SonarClientError("issue belongs to a different Sonar project")
        relative = component.removeprefix(f"{project_key}:")
        file = None if component == project_key else Path(relative)
        if file is not None and (file.is_absolute() or ".." in file.parts):
            raise SonarClientError("malformed Sonar issue file path")
        return SonarIssue(
            as_text(raw.get("key")),
            as_text(raw.get("rule")),
            severity,
            issue_type,
            file,
        )

    def measures(self, project_key: str, *, empty: bool = False) -> SonarMeasures:
        """Parse current measures; missing required metrics are errors for code."""
        names = tuple(SonarMeasures.model_fields)
        payload = self.request(
            "api/measures/component",
            {"component": project_key, "metricKeys": ",".join(names)},
        )
        raw = as_list(as_object(payload.get("component")).get("measures"))
        values = {
            as_text(as_object(m).get("metric")): as_object(m).get("value") for m in raw
        }
        if len(values) != len(raw):
            raise SonarClientError("malformed Sonar response: duplicate metric values")
        if empty or values.get("ncloc") in ("0", 0, 0.0):
            for name in ("ncloc", "cognitive_complexity", "duplicated_lines_density"):
                values.setdefault(name, 0)
        try:
            return SonarMeasures.model_validate(values)
        except ValidationError as exc:
            raise SonarClientError(f"malformed/missing Sonar measures: {exc}") from exc

    def profile(self, project_key: str, *, empty: bool = False) -> str:
        """Serialize Python profile identity and update timestamp for provenance."""
        payload = self.request(
            "api/qualityprofiles/search", {"project": project_key, "language": "py"}
        )
        profiles = as_list(payload.get("profiles"))
        if not profiles and not empty:
            raise SonarClientError("no Python quality profile found for project")
        # lastUsed and projectCount change after scans and are not profile identity.
        identities = []
        for item in profiles:
            profile = as_object(item)
            identity = {
                key: profile[key]
                for key in (
                    "key",
                    "name",
                    "language",
                    "rulesUpdatedAt",
                    "activeRuleCount",
                    "parentKey",
                )
                if key in profile
            }
            as_text(identity.get("key"))
            identities.append(identity)
        identities.sort(key=lambda profile: str(profile["key"]))
        return json.dumps(identities, sort_keys=True, separators=(",", ":"))

    def wait_for_task(self, task_id: str) -> str:
        """Wait for CE SUCCESS; FAILED, CANCELED and timeout fail explicitly."""
        deadline = time.monotonic() + self.config.compute_timeout
        while time.monotonic() < deadline:
            task = as_object(self.request("api/ce/task", {"id": task_id}).get("task"))
            status = as_text(task.get("status"))
            if status == "SUCCESS":
                return as_text(task.get("analysisId"))
            if status in {"FAILED", "CANCELED"}:
                raise SonarClientError(
                    f"Compute Engine task {task_id} {status}: "
                    f"{task.get('errorMessage', '')}"
                )
            if status not in {"PENDING", "IN_PROGRESS"}:
                raise SonarClientError(f"unknown Compute Engine status {status}")
            time.sleep(
                min(self.config.poll_interval, max(0, deadline - time.monotonic()))
            )
        raise SonarClientError(
            f"Compute Engine task {task_id} TIMEOUT; inspect Sonar background tasks"
        )

"""SonarQube Web API client for fetching issues and measures.

Supports both live HTTP access to a self-hosted SonarQube instance and
offline ``--from-fixture`` mode for development and testing. All raw API
responses are normalized into :class:`~deltx.scoring.models.SonarIssue` and
:class:`~deltx.scoring.models.SonarMeasures` records.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from deltx.common.exceptions import SonarClientError
from deltx.scoring.models import SonarIssue, SonarMeasures

logger = logging.getLogger(__name__)

# SonarQube hard-caps page size at 500 and total results at 10,000.
_MAX_PAGE_SIZE = 500
_MAX_TOTAL_RESULTS = 10_000

# Retry configuration for transient server errors.
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds


class SonarClient:
    """HTTP client for a self-hosted SonarQube Web API.

    Args:
        base_url: Root URL of the SonarQube instance (e.g. ``http://localhost:9000``).
        token: API token used as the HTTP Basic username (password is empty).
    """

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._session = requests.Session()
        self._session.auth = (token, "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_issues(
        self,
        component_key: str,
        branch: str | None = None,
    ) -> list[SonarIssue]:
        """Fetch all issues for a project component, handling pagination.

        Args:
            component_key: SonarQube project/component key.
            branch: Optional branch name to scope the query.

        Returns:
            Normalized list of :class:`SonarIssue` records.

        Raises:
            SonarClientError: On unrecoverable HTTP or parsing errors.
        """
        all_issues: list[SonarIssue] = []
        page = 1

        while True:
            params: dict[str, Any] = {
                "componentKeys": component_key,
                "ps": _MAX_PAGE_SIZE,
                "p": page,
                "additionalFields": "rules",
            }
            if branch is not None:
                params["branch"] = branch

            data = self._get("/api/issues/search", params=params)

            raw_issues = data.get("issues", [])
            if not isinstance(raw_issues, list):
                logger.warning("Unexpected 'issues' type in response: %s", type(raw_issues))
                break

            for raw in raw_issues:
                try:
                    all_issues.append(_parse_issue(raw))
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed issue: %s — %s", raw, exc)

            # Pagination: check if we've fetched all available issues.
            paging = data.get("paging", {})
            total = int(paging.get("total", 0))
            fetched_so_far = page * _MAX_PAGE_SIZE

            if fetched_so_far >= total or fetched_so_far >= _MAX_TOTAL_RESULTS:
                break

            page += 1

        logger.info(
            "Fetched %d issues for component=%s (total reported=%d)",
            len(all_issues),
            component_key,
            total if "total" in dir() else len(all_issues),
        )
        return all_issues

    def fetch_measures(self, component_key: str) -> SonarMeasures:
        """Fetch project-level code measures.

        Args:
            component_key: SonarQube project/component key.

        Returns:
            A :class:`SonarMeasures` record.

        Raises:
            SonarClientError: On unrecoverable HTTP or parsing errors.
        """
        metric_keys = "ncloc,complexity,cognitive_complexity,duplicated_lines_density"
        data = self._get(
            "/api/measures/component",
            params={"component": component_key, "metricKeys": metric_keys},
        )

        measures_list = (
            data.get("component", {}).get("measures", [])
        )

        result: dict[str, Any] = {}
        for m in measures_list:
            key = m.get("metric", "")
            value = m.get("value", "0")
            result[key] = value

        return SonarMeasures(
            ncloc=int(result.get("ncloc", 0)),
            complexity=int(result.get("complexity", 0)),
            cognitive_complexity=int(result.get("cognitive_complexity", 0)),
            duplicated_lines_density=float(result.get("duplicated_lines_density", 0.0)),
        )

    # ------------------------------------------------------------------
    # Fixture mode
    # ------------------------------------------------------------------

    @staticmethod
    def from_fixture(issues_path: Path, measures_path: Path | None = None) -> tuple[list[SonarIssue], SonarMeasures | None]:
        """Load issues (and optionally measures) from saved JSON fixtures.

        Args:
            issues_path: Path to a JSON file containing a SonarQube issues response.
            measures_path: Optional path to a JSON file containing measures.

        Returns:
            Tuple of (issues list, measures or None).
        """
        with open(issues_path) as f:
            issues_data = json.load(f)

        raw_issues = issues_data.get("issues", issues_data)
        if isinstance(raw_issues, dict):
            raw_issues = raw_issues.get("issues", [])

        issues = []
        for raw in raw_issues:
            try:
                issues.append(_parse_issue(raw))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed fixture issue: %s — %s", raw, exc)

        measures = None
        if measures_path is not None and measures_path.exists():
            with open(measures_path) as f:
                measures_data = json.load(f)
            component = measures_data.get("component", measures_data)
            measures_list = component.get("measures", []) if isinstance(component, dict) else []
            result: dict[str, Any] = {}
            for m in measures_list:
                result[m.get("metric", "")] = m.get("value", "0")
            measures = SonarMeasures(
                ncloc=int(result.get("ncloc", 0)),
                complexity=int(result.get("complexity", 0)),
                cognitive_complexity=int(result.get("cognitive_complexity", 0)),
                duplicated_lines_density=float(result.get("duplicated_lines_density", 0.0)),
            )

        logger.info("Loaded %d issues from fixture %s", len(issues), issues_path)
        return issues, measures

    @staticmethod
    def save_fixture(data: dict[str, Any], path: Path) -> None:
        """Save a raw SonarQube API response as a JSON fixture.

        Args:
            data: The raw JSON response dict.
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved fixture to %s", path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a GET request with retry on 5xx errors.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            SonarClientError: After exhausting retries or on 4xx errors.
        """
        url = f"{self._base_url}{endpoint}"

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise SonarClientError(
                        f"Request to {url} failed after {_MAX_RETRIES} retries: {exc}"
                    ) from exc
                _backoff(attempt)
                continue

            if resp.status_code >= 500:
                logger.warning(
                    "SonarQube returned %d for %s (attempt %d/%d)",
                    resp.status_code, url, attempt, _MAX_RETRIES,
                )
                if attempt == _MAX_RETRIES:
                    raise SonarClientError(
                        f"SonarQube returned {resp.status_code} for {url} "
                        f"after {_MAX_RETRIES} retries"
                    )
                _backoff(attempt)
                continue

            if resp.status_code >= 400:
                raise SonarClientError(
                    f"SonarQube returned {resp.status_code} for {url}: {resp.text}"
                )

            try:
                return resp.json()  # type: ignore[no-any-return]
            except (json.JSONDecodeError, ValueError) as exc:
                raise SonarClientError(
                    f"Invalid JSON from {url}: {exc}"
                ) from exc

        # Should never reach here, but satisfy the type checker.
        raise SonarClientError(f"Unexpected retry exhaustion for {url}")  # pragma: no cover


def _parse_issue(raw: dict[str, Any]) -> SonarIssue:
    """Parse a raw SonarQube issue JSON object into a ``SonarIssue``."""
    return SonarIssue(
        rule=raw["rule"],
        severity=raw.get("severity", "INFO"),
        type=raw.get("type", "CODE_SMELL"),
        component=raw.get("component", ""),
        line=int(raw.get("line", 0)),
        effort=raw.get("effort", "0min"),
        message=raw.get("message", ""),
    )


def _backoff(attempt: int) -> None:
    """Exponential backoff sleep."""
    delay = _BACKOFF_BASE * (2 ** (attempt - 1))
    logger.debug("Backing off %.1fs before retry", delay)
    time.sleep(delay)

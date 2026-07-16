"""Tests for the SonarQube client (sonar_client.py).

Covers: pagination boundaries, 5xx retry/backoff, malformed JSON,
empty result sets, and fixture mode round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from deltx.common.exceptions import SonarClientError
from deltx.scoring.models import SonarIssue, SonarMeasures
from deltx.scoring.sonar_client import SonarClient, _MAX_PAGE_SIZE


class TestFetchIssuesPagination:
    """Pagination boundary tests for fetch_issues."""

    def _make_response(self, issues: list[dict], total: int, page: int) -> MagicMock:
        """Create a mock response with proper JSON."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "paging": {"pageIndex": page, "pageSize": _MAX_PAGE_SIZE, "total": total},
            "issues": issues,
        }
        return resp

    def _make_issue_dict(self, key: str) -> dict:
        return {
            "key": key,
            "rule": "python:S1234",
            "severity": "MAJOR",
            "type": "CODE_SMELL",
            "component": "src/mod.py",
            "line": 1,
        }

    def test_single_page_exactly_500(self) -> None:
        """500 issues should be fetched in a single page."""
        client = SonarClient("http://localhost:9000", "token")
        issues_data = [self._make_issue_dict(f"K{i}") for i in range(500)]

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = self._make_response(issues_data, total=500, page=1)
            result = client.fetch_issues("project")

        assert len(result) == 500
        assert mock_session.get.call_count == 1

    def test_two_pages_501_issues(self) -> None:
        """501 issues should require two pages."""
        client = SonarClient("http://localhost:9000", "token")
        page1 = [self._make_issue_dict(f"K{i}") for i in range(500)]
        page2 = [self._make_issue_dict("K500")]

        with patch.object(client, "_session") as mock_session:
            mock_session.get.side_effect = [
                self._make_response(page1, total=501, page=1),
                self._make_response(page2, total=501, page=2),
            ]
            result = client.fetch_issues("project")

        assert len(result) == 501
        assert mock_session.get.call_count == 2

    def test_empty_result_set(self) -> None:
        """Empty result should return an empty list."""
        client = SonarClient("http://localhost:9000", "token")

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = self._make_response([], total=0, page=1)
            result = client.fetch_issues("project")

        assert result == []


class TestRetryBackoff:
    """Tests for 5xx retry and backoff behavior."""

    @patch("deltx.scoring.sonar_client._backoff")
    def test_retries_on_500(self, mock_backoff: MagicMock) -> None:
        """Should retry on 5xx status codes."""
        client = SonarClient("http://localhost:9000", "token")

        fail_resp = MagicMock()
        fail_resp.status_code = 500

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "paging": {"total": 0},
            "issues": [],
        }

        with patch.object(client, "_session") as mock_session:
            mock_session.get.side_effect = [fail_resp, success_resp]
            result = client.fetch_issues("project")

        assert result == []
        assert mock_backoff.call_count == 1

    @patch("deltx.scoring.sonar_client._backoff")
    def test_exhausted_retries_raises(self, mock_backoff: MagicMock) -> None:
        """Should raise SonarClientError after exhausting retries."""
        client = SonarClient("http://localhost:9000", "token")

        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.text = "Service Unavailable"

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = fail_resp
            with pytest.raises(SonarClientError, match="503"):
                client.fetch_issues("project")

    def test_4xx_raises_immediately(self) -> None:
        """4xx errors should raise immediately without retry."""
        client = SonarClient("http://localhost:9000", "token")

        fail_resp = MagicMock()
        fail_resp.status_code = 403
        fail_resp.text = "Forbidden"

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = fail_resp
            with pytest.raises(SonarClientError, match="403"):
                client.fetch_issues("project")


class TestMalformedJSON:
    """Tests for malformed response handling."""

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON response should raise SonarClientError."""
        client = SonarClient("http://localhost:9000", "token")

        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = json.JSONDecodeError("err", "doc", 0)

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = resp
            with pytest.raises(SonarClientError, match="Invalid JSON"):
                client.fetch_issues("project")

    def test_malformed_issue_skipped(self) -> None:
        """Issues missing required fields should be skipped."""
        client = SonarClient("http://localhost:9000", "token")

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "paging": {"total": 2},
            "issues": [
                {"rule": "python:S1234", "severity": "MAJOR", "type": "BUG"},  # valid
                {"no_rule": True},  # malformed
            ],
        }

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = resp
            result = client.fetch_issues("project")

        assert len(result) == 1


class TestFetchMeasures:
    """Tests for fetch_measures."""

    def test_basic_measures(self) -> None:
        """Should parse measures correctly."""
        client = SonarClient("http://localhost:9000", "token")

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": "1000"},
                    {"metric": "complexity", "value": "50"},
                    {"metric": "cognitive_complexity", "value": "25"},
                    {"metric": "duplicated_lines_density", "value": "2.5"},
                ],
            },
        }

        with patch.object(client, "_session") as mock_session:
            mock_session.get.return_value = resp
            measures = client.fetch_measures("project")

        assert measures.ncloc == 1000
        assert measures.complexity == 50
        assert measures.cognitive_complexity == 25
        assert measures.duplicated_lines_density == 2.5


class TestFixtureMode:
    """Tests for fixture loading and saving."""

    def test_load_from_fixture(self, sample_issues_path: Path) -> None:
        """Should load issues from the sample fixture file."""
        issues, measures = SonarClient.from_fixture(sample_issues_path)

        assert len(issues) == 12
        assert all(isinstance(i, SonarIssue) for i in issues)
        assert measures is None

    def test_load_with_measures(
        self, sample_issues_path: Path, sample_measures_path: Path
    ) -> None:
        """Should load both issues and measures."""
        issues, measures = SonarClient.from_fixture(
            sample_issues_path, sample_measures_path
        )

        assert len(issues) == 12
        assert measures is not None
        assert measures.ncloc == 1250

    def test_save_and_reload(self, tmp_path: Path) -> None:
        """Save → reload should round-trip correctly."""
        data = {
            "issues": [
                {"rule": "python:S1234", "severity": "MAJOR", "type": "BUG", "component": "a.py"},
            ],
        }
        fixture_path = tmp_path / "test.json"
        SonarClient.save_fixture(data, fixture_path)

        issues, _ = SonarClient.from_fixture(fixture_path)
        assert len(issues) == 1
        assert issues[0].rule == "python:S1234"

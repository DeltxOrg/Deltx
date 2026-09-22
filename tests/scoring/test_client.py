"""No network: mock JSON boundaries and HTTP responses independently."""

import io
import json
from email.message import Message
from http.client import HTTPMessage
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from deltx.common.exceptions import (
    ConfigurationError,
    SonarClientError,
    SonarConnectionRefusedError,
)
from deltx.scoring.models import Severity
from deltx.scoring.sonarqube.client import (
    SonarQubeClient,
    _RejectRedirects,
    as_list,
    as_object,
    as_text,
)
from deltx.scoring.sonarqube.config import SonarConfig


class FakeClient(SonarQubeClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(
            SonarConfig(page_size=1, compute_timeout=0.01, poll_interval=0.001)
        )
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def request(
        self, endpoint: str, params: dict[str, str] | None = None
    ) -> dict[str, object]:
        self.calls.append((endpoint, params))
        return self.responses.pop(0)


def raw_issue(key: str = "a", severity: str = "MAJOR") -> dict[str, object]:
    return {
        "key": key,
        "rule": "python:S1",
        "component": "project:pkg/a.py",
        "type": "BUG",
        "severity": severity,
    }


def page(items: list[object], total: int, field: str = "issues") -> dict[str, object]:
    return {"paging": {"total": total}, field: items}


def test_pagination_and_parsing() -> None:
    client = FakeClient(
        [page([], 2), page([raw_issue()], 2), page([raw_issue("b")], 2)]
    )
    issues = client.issues("project")
    assert [i.key for i in issues] == ["a", "b"]
    assert issues[0].file is not None and issues[0].file.as_posix() == "pkg/a.py"
    assert issues[0].severity == Severity.MAJOR
    assert client.calls[-1][1] == {
        "componentKeys": "project",
        "resolved": "false",
        "s": "FILE_LINE",
        "p": "2",
        "ps": "1",
    }


def test_measures_optional_and_empty() -> None:
    items = [
        {"metric": key, "value": value}
        for key, value in {
            "ncloc": "100",
            "cognitive_complexity": "10",
            "duplicated_lines_density": "2.5",
            "sqale_index": "60",
            "sqale_debt_ratio": "1.1",
        }.items()
    ]
    result = FakeClient([{"component": {"measures": items}}]).measures("project")
    assert result.ncloc == 100 and result.sqale_index == 60
    assert result.duplicated_lines_density == 2.5
    assert (
        FakeClient([{"component": {"measures": []}}])
        .measures("project", empty=True)
        .ncloc
        == 0
    )
    with pytest.raises(SonarClientError):
        FakeClient([{"component": {"measures": []}}]).measures("project")


@pytest.mark.parametrize(
    "payload", [{}, {"paging": {"total": "x"}}, {"paging": {"total": True}}]
)
def test_malformed_pagination(payload: dict[str, object]) -> None:
    with pytest.raises(SonarClientError):
        FakeClient([payload]).issues("project")


def test_unsupported_severity_and_unknown_type() -> None:
    with pytest.raises(ConfigurationError, match="unsupported Sonar severity"):
        SonarQubeClient._issue(raw_issue(severity="HIGH"), "project")
    raw = {**raw_issue(), "type": "OTHER"}
    assert SonarQubeClient._issue(raw, "project").issue_type.value == "UNKNOWN"
    missing = raw_issue()
    del missing["severity"]
    with pytest.raises(ConfigurationError, match="no legacy severity"):
        SonarQubeClient._issue(missing, "project")
    with pytest.raises(SonarClientError, match="different Sonar project"):
        SonarQubeClient._issue(raw_issue(), "other-project")


@pytest.mark.parametrize("status", ["FAILED", "CANCELED", "unexpected"])
def test_compute_failure(status: str) -> None:
    with pytest.raises(SonarClientError):
        FakeClient([{"task": {"status": status}}]).wait_for_task("task")


def test_compute_success() -> None:
    client = FakeClient(
        [
            {"task": {"status": "IN_PROGRESS"}},
            {"task": {"status": "SUCCESS", "analysisId": "done"}},
        ]
    )
    assert client.wait_for_task("task") == "done"


def test_compute_timeout() -> None:
    with patch("deltx.scoring.sonarqube.client.time.monotonic", side_effect=[0, 1]):
        with pytest.raises(SonarClientError, match="TIMEOUT"):
            FakeClient([]).wait_for_task("task")


def test_http_success_and_errors() -> None:
    response = MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps({"status": "UP"}).encode())
    with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
        opened.return_value.open.return_value = response
        assert SonarQubeClient(SonarConfig()).status()["status"] == "UP"
        assert opened.return_value.open.call_args.kwargs["timeout"] == 30
    errors = [
        URLError("offline"),
        TimeoutError("timeout"),
        HTTPError("http://localhost", 401, "no", Message(), None),
    ]
    for error in errors:
        with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
            opened.return_value.open.side_effect = error
            with pytest.raises(SonarClientError):
                SonarQubeClient(SonarConfig()).status()
    response.__enter__.return_value = io.BytesIO(b"not json")
    with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
        opened.return_value.open.return_value = response
        with pytest.raises(SonarClientError, match="malformed"):
            SonarQubeClient(SonarConfig()).status()


def test_refused_connection_and_redirect_are_distinct_errors() -> None:
    with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
        opened.return_value.open.side_effect = URLError(ConnectionRefusedError())
        with pytest.raises(SonarConnectionRefusedError):
            SonarQubeClient(SonarConfig()).status()
    request = Request("http://localhost:19000/api/system/status")  # noqa: S310
    with pytest.raises(SonarClientError, match="redirect"):
        _RejectRedirects().redirect_request(
            request, io.BytesIO(), 302, "Found", HTTPMessage(), "http://elsewhere/"
        )


def test_json_narrowing() -> None:
    for function, value in [(as_list, {}), (as_object, []), (as_text, "")]:
        with pytest.raises(SonarClientError):
            function(value)


def test_incomplete_duplicate_and_overflow_pages() -> None:
    for responses in [
        [page([], 1), page([], 1)],
        [page([], 2), page([raw_issue()], 2), page([raw_issue()], 2)],
        [page([], 2), page([raw_issue()], 2), page([raw_issue("b")], 3)],
    ]:
        with pytest.raises(SonarClientError):
            FakeClient(responses).issues("project")
    with pytest.raises(SonarClientError, match="10,000"):
        FakeClient([page([], 10001)])._pages("api/issues/search", {}, "issues")


def test_profile() -> None:
    profile = {"key": "python-default", "name": "Sonar way", "rulesUpdatedAt": "date"}
    assert "python-default" in FakeClient([{"profiles": [profile]}]).profile("project")
    with pytest.raises(SonarClientError):
        FakeClient([{"profiles": []}]).profile("project")
    first = FakeClient(
        [{"profiles": [{**profile, "lastUsed": "yesterday", "projectCount": 1}]}]
    )
    second = FakeClient(
        [{"profiles": [{**profile, "lastUsed": "today", "projectCount": 2}]}]
    )
    assert first.profile("project") == second.profile("project")


def test_duplicate_metrics_are_rejected() -> None:
    duplicated = [{"metric": "ncloc", "value": "0"}] * 2
    with pytest.raises(SonarClientError, match="duplicate metric"):
        FakeClient([{"component": {"measures": duplicated}}]).measures("project")


def test_partition_above_issue_search_cap() -> None:
    first = [raw_issue(str(i)) for i in range(6000)]
    second = [raw_issue(str(i)) for i in range(6000, 10001)]
    client = FakeClient(
        [
            page([], 10001),
            page(
                [{"key": "project:one.py"}, {"key": "project:two.py"}], 2, "components"
            ),
            page(list(first), 6000),
            page(list(second), 4001),
        ]
    )
    assert len(client.issues("project")) == 10001
    broken = FakeClient(
        [
            page([], 10001),
            page([], 0, "components"),
        ]
    )
    with pytest.raises(SonarClientError, match="partition totals differ"):
        broken.issues("project")


def test_empty_state_metrics_and_profile() -> None:
    result = FakeClient(
        [{"component": {"measures": [{"metric": "ncloc", "value": "0"}]}}]
    ).measures("p")
    assert result.cognitive_complexity == result.duplicated_lines_density == 0
    assert FakeClient([{"profiles": []}]).profile("p", empty=True) == "[]"

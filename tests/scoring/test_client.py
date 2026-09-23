"""No network: mock JSON boundaries and HTTP responses independently."""

import io
import json
from collections.abc import MutableMapping
from dataclasses import FrozenInstanceError
from email.message import Message
from http.client import HTTPMessage
from itertools import permutations
from typing import cast
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from pydantic import SecretStr

from deltx.common.exceptions import (
    ConfigurationError,
    SonarClientError,
    SonarConnectionRefusedError,
)
from deltx.scoring.models import (
    CleanCodeAttribute,
    Dimension,
    RuleCatalog,
    Severity,
    SonarRuleMetadata,
)
from deltx.scoring.sonarqube.client import (
    SonarQubeClient,
    _RejectRedirects,
    as_list,
    as_object,
    as_text,
)
from deltx.scoring.sonarqube.config import SonarConfig


def config() -> SonarConfig:
    return SonarConfig(
        _env_file=None,
        host_url="http://localhost:19876/sonar",
        token=SecretStr("unit-test-token"),
    )


class FakeClient(SonarQubeClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(
            config().model_copy(
                update={"page_size": 1, "compute_timeout": 0.01, "poll_interval": 0.001}
            )
        )
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def request(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
        *,
        authenticated: bool = True,
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
        "components": "project",
        "issueStatuses": "OPEN,CONFIRMED",
        "s": "FILE_LINE",
        "p": "2",
        "ps": "1",
    }


def test_measures_required_and_optional_debt_ratio() -> None:
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
    items.pop()  # Only debt ratio is optional; remediation effort is required.
    assert (
        FakeClient([{"component": {"measures": items}}])
        .measures("project")
        .sqale_debt_ratio
        is None
    )
    with pytest.raises(SonarClientError, match="missing Sonar measures"):
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
    with pytest.raises(ConfigurationError, match="no severity or impacts"):
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
        assert SonarQubeClient(config()).status()["status"] == "UP"
        assert opened.return_value.open.call_args.kwargs["timeout"] == 30
        request = opened.return_value.open.call_args.args[0]
        assert request.full_url == "http://localhost:19876/sonar/api/system/status?"
        assert request.get_header("Authorization") is None
    errors = [
        URLError("offline"),
        TimeoutError("timeout"),
        HTTPError("http://localhost", 401, "no", Message(), None),
    ]
    for error in errors:
        with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
            opened.return_value.open.side_effect = error
            with pytest.raises(SonarClientError):
                SonarQubeClient(config()).status()
    response.__enter__.return_value = io.BytesIO(b"not json")
    with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
        opened.return_value.open.return_value = response
        with pytest.raises(SonarClientError, match="malformed"):
            SonarQubeClient(config()).status()


def test_refused_connection_and_redirect_are_distinct_errors() -> None:
    with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
        opened.return_value.open.side_effect = URLError(ConnectionRefusedError())
        with pytest.raises(SonarConnectionRefusedError):
            SonarQubeClient(config()).status()
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


def test_empty_state_does_not_invent_metrics_or_rule_coverage() -> None:
    with pytest.raises(SonarClientError, match="missing Sonar measures"):
        FakeClient(
            [{"component": {"measures": [{"metric": "ncloc", "value": "0"}]}}]
        ).measures("p")
    assert FakeClient([{"profiles": []}]).profile("p", empty=True) == "[]"
    with pytest.raises(ConfigurationError, match="no active Python quality profile"):
        FakeClient([]).rule_catalog("[]")


@pytest.mark.parametrize("ncloc", ["0", "100"])
@pytest.mark.parametrize(
    "missing", ["sqale_index", "cognitive_complexity", "duplicated_lines_density"]
)
def test_required_metrics_fail_even_when_ncloc_is_zero(
    missing: str, ncloc: str
) -> None:
    values = {
        "ncloc": ncloc,
        "sqale_index": "0",
        "cognitive_complexity": "0",
        "duplicated_lines_density": "0",
    }
    items = [{"metric": key, "value": value} for key, value in values.items()]
    measured = FakeClient([{"component": {"measures": items}}]).measures("p")
    assert (
        measured.sqale_index
        == measured.cognitive_complexity
        == measured.duplicated_lines_density
        == 0
    )
    del values[missing]
    items = [{"metric": key, "value": value} for key, value in values.items()]
    with pytest.raises(SonarClientError, match=missing):
        FakeClient([{"component": {"measures": items}}]).measures("p")


@pytest.mark.parametrize(
    "level,expected",
    [
        ("INFO", Severity.INFO),
        ("LOW", Severity.MINOR),
        ("MEDIUM", Severity.MAJOR),
        ("HIGH", Severity.CRITICAL),
        ("BLOCKER", Severity.BLOCKER),
    ],
)
def test_mqr_issues_without_legacy_fields(level: str, expected: Severity) -> None:
    raw = raw_issue()
    del raw["severity"], raw["type"]
    raw["impacts"] = [{"softwareQuality": "RELIABILITY", "severity": level}]
    issue = SonarQubeClient._issue(raw, "project")
    assert issue.severity == expected
    assert issue.impacts[0].dimension == Dimension.CORRECTNESS
    assert issue.impacts[0].severity == expected
    assert (
        issue.severity.weight
        == ["INFO", "LOW", "MEDIUM", "HIGH", "BLOCKER"].index(level) + 1
    )


def test_mqr_preserves_each_severity_and_counts_issue_once_at_maximum() -> None:
    raw = raw_issue(severity="BLOCKER")
    raw["impacts"] = [
        {"softwareQuality": "SECURITY", "severity": "HIGH"},
        {"softwareQuality": "MAINTAINABILITY", "severity": "LOW"},
    ]
    issue = SonarQubeClient._issue(raw, "project")
    assert issue.severity == Severity.CRITICAL  # MQR wins over legacy severity.
    assert {i.dimension: i.severity for i in issue.impacts} == {
        Dimension.SECURITY: Severity.CRITICAL,
        Dimension.MAINTAINABILITY: Severity.MINOR,
    }


@pytest.mark.parametrize(
    "impacts,error",
    [
        ([{"softwareQuality": "SECURITY", "severity": "EXTREME"}], ConfigurationError),
        ([{"softwareQuality": "OTHER", "severity": "HIGH"}], ConfigurationError),
        ([{"softwareQuality": "EFFICIENCY", "severity": "HIGH"}], ConfigurationError),
        ([{"softwareQuality": "SECURITY"}], SonarClientError),
        (None, SonarClientError),
    ],
)
def test_bad_mqr_impacts_fail_instead_of_losing_evidence(
    impacts: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        SonarQubeClient._issue({**raw_issue(), "impacts": impacts}, "project")


def test_mqr_debt_wins_over_legacy_metric() -> None:
    values = {
        "ncloc": "1",
        "cognitive_complexity": "0",
        "duplicated_lines_density": "0",
        "sqale_index": "10",
        "software_quality_maintainability_remediation_effort": "45",
        "software_quality_maintainability_debt_ratio": "12.5",
    }
    client = FakeClient(
        [
            {
                "component": {
                    "measures": [
                        {"metric": key, "value": value} for key, value in values.items()
                    ]
                }
            }
        ]
    )
    result = client.measures("project")
    assert result.sqale_index == 45
    assert result.sqale_debt_ratio == 12.5
    assert "software_quality_maintainability_remediation_effort" in str(client.calls)


def test_api_bearer_auth_uses_configured_context_and_port() -> None:
    response = MagicMock()
    response.__enter__.return_value = io.BytesIO(b'{"valid":true}')
    with patch("deltx.scoring.sonarqube.client.build_opener") as opened:
        opened.return_value.open.return_value = response
        SonarQubeClient(config()).request("api/authentication/validate")
        request = opened.return_value.open.call_args.args[0]
        assert request.get_header("Authorization") == "Bearer unit-test-token"
        assert request.full_url.startswith(config().host_url + "/api/")


def test_duplicate_modern_impacts_reduce_to_strongest_in_any_order() -> None:
    impacts = [
        {"softwareQuality": "RELIABILITY", "severity": "MEDIUM"},
        {"softwareQuality": "RELIABILITY", "severity": "HIGH"},
        {"softwareQuality": "MAINTAINABILITY", "severity": "LOW"},
    ]
    for ordering in permutations(impacts):
        issue = SonarQubeClient._issue(
            {**raw_issue(), "impacts": list(ordering)}, "project"
        )
        assert issue.severity == Severity.CRITICAL
        assert {i.dimension: i.severity for i in issue.impacts} == {
            Dimension.CORRECTNESS: Severity.CRITICAL,
            Dimension.MAINTAINABILITY: Severity.MINOR,
        }
        assert len(issue.impacts) == 2


@pytest.mark.parametrize("severity", list(Severity))
def test_legacy_severity_weights(severity: Severity) -> None:
    issue = SonarQubeClient._issue(raw_issue(severity=severity.value), "project")
    assert issue.severity.weight == list(Severity).index(severity) + 1


def raw_rule(key: str = "python:S1", attribute: str = "EFFICIENT") -> dict[str, object]:
    return {
        "key": key,
        "lang": "py",
        "cleanCodeAttribute": attribute,
        "impacts": [{"softwareQuality": "RELIABILITY", "severity": "HIGH"}],
    }


def test_rule_catalog_pages_once_and_refreshes_after_profile_change() -> None:
    pages = [
        {"total": 2, "rules": [raw_rule()]},
        {"total": 2, "rules": [raw_rule("python:S2", "LOGICAL")]},
    ]
    client = FakeClient(pages * 2)
    profile = '[{"key":"py","activeRuleCount":2,"rulesUpdatedAt":"before"}]'
    catalog = client.rule_catalog(profile)
    assert catalog.efficiency_rule_keys == ("python:S1",)
    assert catalog.get("python:S1").clean_code_attribute is CleanCodeAttribute.EFFICIENT
    assert catalog.get("python:S2").impacts[0].severity is Severity.CRITICAL
    for _ in range(10):
        assert client.rule_catalog(profile) is catalog
    assert len(client.calls) == 2
    assert client.calls[0] == (
        "api/rules/search",
        {
            "qprofile": "py",
            "activation": "true",
            "languages": "py",
            "p": "1",
            "ps": "1",
        },
    )
    assert client.rule_catalog(profile.replace("before", "after")) == catalog
    assert len(client.calls) == 4


def test_rules_normalize_once_at_api_boundary() -> None:
    raw = raw_rule(attribute=" efficient ")
    raw["impacts"] = [{"softwareQuality": " reliability ", "severity": " high "}]
    rule = SonarQubeClient._rule(raw)
    assert rule.clean_code_attribute is CleanCodeAttribute.EFFICIENT
    assert rule.impacts[0].dimension is Dimension.CORRECTNESS
    assert rule.impacts[0].severity is Severity.CRITICAL


@pytest.mark.parametrize("attribute", list(CleanCodeAttribute))
def test_all_supported_clean_code_attributes(attribute: CleanCodeAttribute) -> None:
    assert (
        SonarQubeClient._rule(raw_rule(attribute=attribute.value)).clean_code_attribute
        is attribute
    )


def test_invalid_and_missing_rule_metadata_fail() -> None:
    missing = raw_rule()
    del missing["cleanCodeAttribute"]
    with pytest.raises(SonarClientError, match="missing cleanCodeAttribute"):
        SonarQubeClient._rule(missing)
    with pytest.raises(ConfigurationError, match="unsupported cleanCodeAttribute"):
        SonarQubeClient._rule(raw_rule(attribute="PERFORMANCE"))
    with pytest.raises(SonarClientError, match="Python rule metadata"):
        SonarQubeClient._rule({**raw_rule(), "lang": "java"})
    with pytest.raises(SonarClientError, match="malformed Python profile"):
        FakeClient([]).rule_catalog("not JSON")


def test_no_active_efficiency_rules_fails_instead_of_perfect_score() -> None:
    client = FakeClient([page([raw_rule(attribute="LOGICAL")], 1, "rules")])
    with pytest.raises(ConfigurationError, match="no EFFICIENT rules"):
        client.rule_catalog('[{"key":"py","activeRuleCount":1}]')


def test_rule_catalog_rejects_duplicate_and_changed_counts() -> None:
    profile = '[{"key":"py","activeRuleCount":2}]'
    with pytest.raises(SonarClientError, match="duplicate rules"):
        FakeClient([page([raw_rule()], 2, "rules")] * 2).rule_catalog(profile)
    with pytest.raises(SonarClientError, match="count differs"):
        FakeClient([page([raw_rule()], 1, "rules")]).rule_catalog(profile)


def test_rule_catalog_copies_and_freezes_its_metadata() -> None:
    rule = SonarQubeClient._rule(raw_rule())
    original = {rule.key: rule}
    catalog = RuleCatalog(original)
    original.clear()
    assert catalog.get(rule.key) == rule
    with pytest.raises(TypeError):
        cast(MutableMapping[str, SonarRuleMetadata], catalog.rules)["python:other"] = (
            rule
        )
    with pytest.raises(FrozenInstanceError):
        rule.clean_code_attribute = CleanCodeAttribute.LOGICAL  # type: ignore[misc]
    with pytest.raises(ConfigurationError, match="keys do not match"):
        RuleCatalog({"wrong": rule})

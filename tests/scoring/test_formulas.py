"""Formula invariants and deliberately adversarial numeric inputs."""

import math
from collections.abc import MutableMapping
from typing import cast

import pytest
from pydantic import ValidationError

from deltx.common.exceptions import ScoringError
from deltx.scoring.config import (
    DimensionConfig,
    MetricConfig,
    RuleMapping,
    ScoringConfig,
)
from deltx.scoring.models import Dimension, Severity
from deltx.scoring.squale.formulas import (
    aggregate,
    bounded_log,
    density,
    dynamic_weight,
    individual_mark,
    metric_mark,
)


def test_severity_intrinsic_and_order() -> None:
    config = ScoringConfig()
    weights = [
        dynamic_weight(config.severity_values[s], 0, 0, 0, 1, DimensionConfig())
        for s in Severity
    ]
    assert weights == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_context_increases_risk(index: int) -> None:
    context = [0.0, 0.0, 0.0]
    base = dynamic_weight(3, context[0], context[1], context[2], 1, DimensionConfig())
    context[index] = 1
    assert (
        dynamic_weight(3, context[0], context[1], context[2], 1, DimensionConfig())
        > base
    )


@pytest.mark.parametrize("value", [0.0, 0.001, 1, 100, 1e300])
def test_normalization_bounded(value: float) -> None:
    assert 0 <= bounded_log(value) < 1


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_invalid_normalization(value: float) -> None:
    with pytest.raises(ScoringError):
        bounded_log(value)


def test_config_constraints() -> None:
    with pytest.raises(ValidationError, match="must equal 1"):
        DimensionConfig(alpha=0.5)
    for values in ({"rho": -1}, {"kappa": 0}, {"squale_lambda": 1}, {"beta": math.nan}):
        with pytest.raises(ValidationError):
            DimensionConfig.model_validate(values)
    with pytest.raises(ValidationError):
        ScoringConfig(dimensions={})
    with pytest.raises(ValidationError):
        ScoringConfig(severity_values={Severity.BLOCKER: 5})
    for weight in (0, -1, math.inf, math.nan):
        with pytest.raises(ValidationError):
            ScoringConfig(maintainability_issue_omega=weight)


def test_config_is_deeply_immutable_and_serializable() -> None:
    original = {s: float(i) for i, s in enumerate(Severity, 1)}
    config = ScoringConfig(
        severity_values=original,
        rule_overrides=(
            RuleMapping(rule="python:test", coefficients={Dimension.EFFICIENCY: 1}),
        ),
    )
    original[Severity.INFO] = 0.5
    assert config.severity_values[Severity.INFO] == 1
    with pytest.raises(TypeError):
        cast(MutableMapping[Severity, float], config.severity_values)[Severity.INFO] = (
            0.5
        )
    with pytest.raises(TypeError):
        cast(MutableMapping[Dimension, float], config.rule_overrides[0].coefficients)[
            Dimension.EFFICIENCY
        ] = 0.5
    with pytest.raises(ValidationError):
        config.dimensions[Dimension.SECURITY].rho = 2  # type: ignore[misc]
    assert ScoringConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("base", [1.000000000001, 2, 9, 100, 1e300])
def test_squale_endpoints_and_range(base: float) -> None:
    assert aggregate([], base) == 100
    assert aggregate([(3, 1), (3, 2)], base) == pytest.approx(100)
    assert aggregate([(0, 1), (0, 2)], base) == 0
    assert 0 <= aggregate([(0, 1), (3, 1)], base) <= 100


def test_conflict_resistance_and_weights() -> None:
    mixed = [(0.0, 1.0), (3.0, 1.0)]
    assert aggregate(mixed, 9) < 50
    assert aggregate(mixed, 30) < aggregate(mixed, 9)
    assert aggregate([(0, 2), (3, 1)], 9) < aggregate(mixed, 9)
    expected = -math.log((9**0 + 2 * 9**-3) / 3) / math.log(9) * 100 / 3
    assert aggregate([(0, 1), (3, 2)], 9) == pytest.approx(expected)


@pytest.mark.parametrize(
    "marks,base", [([(4, 1)], 9), ([(1, 0)], 9), ([(1, 1)], 1), ([(math.nan, 1)], 9)]
)
def test_invalid_squale(marks: list[tuple[float, float]], base: float) -> None:
    with pytest.raises(ScoringError):
        aggregate(marks, base)


def test_extreme_numeric_stability() -> None:
    assert math.isfinite(aggregate([(0, 1e-300), (3, 1e300)], 1e300))
    assert aggregate([(0, 1), (3, 1)], 1.000000000001) == pytest.approx(50, abs=0.01)
    assert metric_mark(1e300, MetricConfig(tau=1e-300, k=1000)) == 0
    assert metric_mark(0, MetricConfig(tau=1)) == 3
    assert density(0, 0) == 0


def test_individual_and_synthetic_marks() -> None:
    assert individual_mark(10, DimensionConfig()) == 0
    assert individual_mark(0, DimensionConfig()) == 3
    assert metric_mark(10, MetricConfig(tau=10)) == 1.5
    with pytest.raises(ScoringError):
        metric_mark(-1, MetricConfig(tau=1))
    with pytest.raises(ScoringError):
        dynamic_weight(5, 2, 0, 0, 1, DimensionConfig())
    with pytest.raises(ScoringError):
        individual_mark(math.nan, DimensionConfig())

"""Conflict resistance test (Phase 7) — the core research claim.

This test operationalizes the "semantic gap" that the whole architecture
is designed to resolve. It constructs two scenarios:

- **Case A**: 50 MINOR maintainability smells in a peripheral file
  (low centrality, low churn).
- **Case B**: 1 BLOCKER security vulnerability in a high-centrality
  routing module (high centrality, high churn).

The assertion: Case B's system security score must be **strictly lower**
than Case A's maintainability score. This means the system correctly
refuses to average away the critical vulnerability, even when the sheer
*count* of Case A's issues is 50× higher.

If a refactor ever makes these two cases approximately equal, the build
should fail — this is a permanent regression guard.
"""

from __future__ import annotations

import pytest

from deltx.scoring.aggregation import squale_aggregate
from deltx.scoring.models import (
    Hyperparams,
    IsoDimension,
    SonarIssue,
    WeightedIssue,
)
from deltx.scoring.scoring import Normalizer, module_score
from deltx.scoring.weighting import weight_all


@pytest.fixture
def hp() -> Hyperparams:
    """Default hyperparameters for conflict resistance testing."""
    return Hyperparams(alpha=0.5, beta=1.0, gamma=0.3, lam=30.0)


@pytest.fixture
def normalizer() -> Normalizer:
    """A normalizer fitted on a reasonably wide range."""
    n = Normalizer()
    n.fit({
        IsoDimension.MAINTAINABILITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        IsoDimension.CORRECTNESS: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        IsoDimension.SECURITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        IsoDimension.EFFICIENCY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    })
    return n


class TestConflictResistance:
    """The golden property test: the system must not average away critical defects."""

    def test_blocker_in_core_dominates_minor_swarm(
        self, hp: Hyperparams, normalizer: Normalizer
    ) -> None:
        """1 BLOCKER in a central module must produce a worse system score
        than 50 MINOR smells in a peripheral file.

        This is the test that validates the entire research claim.
        """
        # --- Case A: 50 MINOR maintainability smells in a peripheral file ---
        case_a_issues = [
            SonarIssue(
                rule="python:S1192",
                severity="MINOR",
                type="CODE_SMELL",
                component="src/utils/helpers.py",
                line=i,
            )
            for i in range(50)
        ]

        # Peripheral file: low centrality (0.05), low churn (5 lines).
        case_a_weighted = weight_all(
            case_a_issues,
            centrality_fn=lambda _: 0.05,
            churn_fn=lambda _: 5.0,
            hp=hp,
        )

        # Score Case A as a module.
        case_a_scores = module_score(case_a_weighted, loc=500, normalizer=normalizer)

        # --- Case B: 1 BLOCKER security vulnerability in a central router ---
        case_b_issues = [
            SonarIssue(
                rule="python:S5542",
                severity="BLOCKER",
                type="VULNERABILITY",
                component="src/core/router.py",
                line=1,
            )
        ]

        # Central file: high centrality (0.95), high churn (200 lines).
        case_b_weighted = weight_all(
            case_b_issues,
            centrality_fn=lambda _: 0.95,
            churn_fn=lambda _: 200.0,
            hp=hp,
        )

        # Score Case B as a module.
        case_b_scores = module_score(case_b_weighted, loc=200, normalizer=normalizer)

        # --- The assertion ---
        # Case B's security score (the dimension hit by the BLOCKER) must be
        # strictly lower than Case A's maintainability score (the dimension
        # hit by the 50 MINORs).
        assert case_b_scores.security < case_a_scores.maintainability, (
            f"CONFLICT RESISTANCE FAILURE:\n"
            f"  Case A (50 MINOR smells): maintainability = {case_a_scores.maintainability:.2f}\n"
            f"  Case B (1 BLOCKER vuln):  security = {case_b_scores.security:.2f}\n"
            f"  The system is averaging away critical defects!"
        )

    def test_squale_aggregation_does_not_mask_critical(
        self, hp: Hyperparams, normalizer: Normalizer
    ) -> None:
        """System-level Squale aggregation must not mask a single bad module.

        Even with multiple healthy modules, one catastrophic module must
        drag the system score well below the arithmetic mean.
        """
        # 4 healthy modules + 1 catastrophic module.
        healthy_scores = [92.0, 95.0, 88.0, 91.0]
        catastrophic = 15.0

        all_scores = healthy_scores + [catastrophic]
        arithmetic_mean = sum(all_scores) / len(all_scores)

        agg = squale_aggregate(all_scores, lam=hp.lam)

        # Squale aggregate must be far below the arithmetic mean.
        assert agg < arithmetic_mean - 15, (
            f"System aggregate {agg:.1f} too close to arithmetic mean {arithmetic_mean:.1f}"
        )

        # The aggregate should be well below the arithmetic mean (under 75%).
        assert agg < arithmetic_mean * 0.75, (
            f"System aggregate {agg:.1f} not pulled down enough by catastrophic module "
            f"(should be < {arithmetic_mean * 0.75:.1f})"
        )

    def test_arithmetic_mean_would_fail(self) -> None:
        """Verify that naive arithmetic mean WOULD mask the critical defect.

        This test exists to confirm the problem the Squale formula solves.
        It does not test our code — it tests the *alternative* we rejected.
        """
        # Same setup as above: 4 healthy + 1 catastrophic.
        all_scores = [92.0, 95.0, 88.0, 91.0, 15.0]
        arithmetic_mean = sum(all_scores) / len(all_scores)

        # The arithmetic mean is high enough to mask the critical module.
        assert arithmetic_mean > 70, (
            f"Arithmetic mean {arithmetic_mean:.1f} should be deceptively high"
        )

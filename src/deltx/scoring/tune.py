"""Hyperparameter tuning script (Phase 7).

Grid-searches α, β, γ, λ against a labeled fixture set, maximizing
Spearman rank correlation between predicted quality ordering and the
expected ordering. Writes the best parameters to a JSON file.

Usage::

    poetry run python -m deltx.scoring.tune \\
        --output data/scoring/hyperparams.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import spearmanr  # type: ignore[import-untyped]

from deltx.scoring.aggregation import squale_aggregate
from deltx.scoring.models import Hyperparams, IsoDimension, SonarIssue
from deltx.scoring.scoring import Normalizer, module_score
from deltx.scoring.weighting import weight_all

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Labeled fixture: modules hand-ranked worst → best
# ---------------------------------------------------------------------------

@dataclass
class LabeledModule:
    """A module with a known quality rank (lower = worse)."""

    name: str
    issues: list[SonarIssue]
    centrality: float
    churn: float
    loc: int
    expected_rank: int  # 1 = worst, higher = better


def _build_labeled_fixtures() -> list[LabeledModule]:
    """Build a small set of hand-ranked modules for tuning.

    The ranking reflects the intuitive severity ordering:
    1. Critical security vulnerability in a central module (worst).
    2. Multiple major bugs in a moderately central file.
    3. Many minor smells in a peripheral utility.
    4. A clean module with minimal issues (best).
    """
    return [
        LabeledModule(
            name="auth_module",
            issues=[
                SonarIssue(rule="python:S5542", severity="BLOCKER",
                           type="VULNERABILITY", component="auth.py"),
                SonarIssue(rule="python:S4790", severity="CRITICAL",
                           type="SECURITY_HOTSPOT", component="auth.py"),
            ],
            centrality=0.9,
            churn=200.0,
            loc=150,
            expected_rank=1,  # worst
        ),
        LabeledModule(
            name="engine_module",
            issues=[
                SonarIssue(rule="python:S1854", severity="MAJOR",
                           type="BUG", component="engine.py"),
                SonarIssue(rule="python:S1854", severity="MAJOR",
                           type="BUG", component="engine.py"),
                SonarIssue(rule="python:S1854", severity="MAJOR",
                           type="BUG", component="engine.py"),
            ],
            centrality=0.6,
            churn=100.0,
            loc=300,
            expected_rank=2,
        ),
        LabeledModule(
            name="utils_module",
            issues=[
                SonarIssue(rule="python:S1192", severity="MINOR",
                           type="CODE_SMELL", component="utils.py")
                for _ in range(20)
            ],
            centrality=0.1,
            churn=10.0,
            loc=500,
            expected_rank=3,
        ),
        LabeledModule(
            name="config_module",
            issues=[
                SonarIssue(rule="python:S1134", severity="INFO",
                           type="CODE_SMELL", component="config.py"),
            ],
            centrality=0.05,
            churn=2.0,
            loc=50,
            expected_rank=4,  # best
        ),
    ]


def _score_modules(
    modules: list[LabeledModule],
    hp: Hyperparams,
    normalizer: Normalizer,
) -> list[float]:
    """Score each module and return a list of composite scores."""
    scores: list[float] = []
    for mod in modules:
        weighted = weight_all(
            mod.issues,
            centrality_fn=lambda _, c=mod.centrality: c,
            churn_fn=lambda _, ch=mod.churn: ch,
            hp=hp,
        )
        dim_scores = module_score(weighted, mod.loc, normalizer)
        # Use the average across dimensions as a composite score.
        vals = list(dim_scores.to_dict().values())
        composite = sum(vals) / len(vals)
        scores.append(composite)
    return scores


def tune(
    output_path: Path | None = None,
) -> Hyperparams:
    """Grid-search hyperparameters maximizing Spearman rank correlation.

    Returns:
        The best-performing ``Hyperparams`` instance.
    """
    modules = _build_labeled_fixtures()
    expected_ranks = [m.expected_rank for m in modules]

    # Build a normalizer fitted on a synthetic range.
    normalizer = Normalizer()
    normalizer.fit({
        IsoDimension.MAINTAINABILITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        IsoDimension.CORRECTNESS: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        IsoDimension.SECURITY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        IsoDimension.EFFICIENCY: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    })

    # Grid search space.
    alpha_grid = [0.3, 0.5, 0.7, 1.0]
    beta_grid = [0.5, 1.0, 1.5, 2.0]
    gamma_grid = [0.1, 0.3, 0.5, 1.0]
    lam_grid = [10.0, 20.0, 30.0, 50.0]

    best_corr = -2.0
    best_hp = Hyperparams()

    for alpha, beta, gamma, lam in itertools.product(
        alpha_grid, beta_grid, gamma_grid, lam_grid
    ):
        hp = Hyperparams(alpha=alpha, beta=beta, gamma=gamma, lam=lam)

        try:
            predicted_scores = _score_modules(modules, hp, normalizer)
        except Exception:
            continue

        # Higher score = better module, so rank them.
        # Spearman correlation between predicted ordering and expected ordering.
        # We want the predicted scores to rank-correlate with expected_ranks.
        # Since expected_rank=1 is worst and score is higher=better,
        # a perfect correlation means the module with rank 4 gets the highest score.
        corr, _ = spearmanr(predicted_scores, expected_ranks)

        if corr > best_corr:
            best_corr = corr
            best_hp = hp

    logger.info(
        "Best hyperparams: α=%.2f, β=%.2f, γ=%.2f, λ=%.1f (Spearman=%.4f)",
        best_hp.alpha, best_hp.beta, best_hp.gamma, best_hp.lam, best_corr,
    )

    if output_path is not None:
        best_hp.to_json(output_path)
        logger.info("Saved to %s", output_path)

    return best_hp


def main() -> None:
    """CLI entry point for the tuning script."""
    parser = argparse.ArgumentParser(
        description="Grid-search scoring hyperparameters"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scoring/hyperparams.json"),
        help="Output path for the tuned hyperparameters JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    tune(output_path=args.output)


if __name__ == "__main__":
    main()

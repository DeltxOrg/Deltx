"""Phase A of the production training routine: assemble a balanced corpus.

Downloads the three usable training sources (AIGCodeSet, DroidCollection, and
CodeNet — GPTSniffer is excluded as it ships zero Python samples), unifies,
filters, and deduplicates them with :class:`DatasetManager`, then downsamples to
a balanced 50/50 human/AI corpus and writes it to
``data/processed/train_balanced.parquet``.

This script performs **no feature extraction and no training**. It only prepares
the raw balanced text corpus. The next steps are:

    Phase B  notebooks/extract_features_gpu.ipynb   (GPU: text -> 16-D features)
    Phase C  scripts/train_detector.py              (local: train + evaluate)

Balancing happens **before** extraction so the expensive language-model pass in
Phase B is never spent on samples that would be discarded to balance the classes.
The target is deliberately over-provisioned (``--per-class`` default 11,000) to
absorb the rows Phase B rejects (unparseable code / a feature family failing);
Phase C rebalances to an exact per-class count afterwards.

Usage::

    # Full build (downloads on first run; reads CodeNet's ~240k files once):
    poetry run python scripts/build_training_set.py

    # Fast smoke test on AIGCodeSet only (no large downloads):
    poetry run python scripts/build_training_set.py --sources aigcodeset --limit 200

Notes:
    ``--max-per-source`` caps each source *before* the Python/length/dedup filters
    run, so a source may end up below its cap. Its only jobs here are to bound
    memory against DroidCollection's ~262k Python rows and to guarantee enough
    supply for the balanced target; the final extraction cost is set by
    ``--per-class``, not by this cap.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltx.common.config import DeltxConfig  # noqa: E402
from deltx.common.exceptions import DeltxError  # noqa: E402
from deltx.detection.dataset import DatasetManager  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# AIGCodeSet is listed first so its supplementary rows win the deduplication
# collision against CodeNet (its human half is a subset of Python800), keeping
# its ``ai_model`` provenance intact where it matters.
DEFAULT_SOURCES = ["aigcodeset", "droidcollection", "codenet"]
DEFAULT_PER_CLASS = 11_000
DEFAULT_MAX_PER_SOURCE = 40_000
DEFAULT_OUTPUT = Path("data/processed/train_balanced.parquet")
HUMAN_LABEL = 0
AI_LABEL = 1

_DOWNLOADERS = {
    "aigcodeset": "download_aigcodeset",
    "droidcollection": "download_droidcollection",
    "codenet": "download_codenet_python",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        choices=DEFAULT_SOURCES,
        help="Which sources to include (default: all three).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_PER_CLASS,
        help="Target rows per class after balancing (over-provisioned).",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=DEFAULT_MAX_PER_SOURCE,
        help="Per-source cap applied before filtering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the balanced parquet.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the raw sources are already present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Smoke-test shortcut: cap each source and set per-class to LIMIT//2 "
            "for a fast end-to-end dry run."
        ),
    )
    return parser.parse_args(argv)


def download_sources(manager: DatasetManager, sources: list[str]) -> None:
    """Download each requested source (idempotent — skips cached data)."""
    for source in sources:
        method = getattr(manager, _DOWNLOADERS[source])
        console.print(f"Downloading [bold]{source}[/bold] (skipped if cached)…")
        method()


def balance(unified: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Downsample ``unified`` to an equal number of human and AI rows.

    The per-class count is clamped to whatever the scarcer class can supply, so a
    thin AI half never forces an error — it just lowers the target.

    Args:
        unified: The unified, filtered frame from :meth:`load_and_unify`.
        per_class: Desired rows per class before clamping.
        seed: Random seed for the class subsampling and the final shuffle.

    Returns:
        A shuffled, class-balanced frame in the unified schema.

    Raises:
        DeltxError: If either class is empty.
    """
    counts = unified["label"].value_counts()
    n_human = int(counts.get(HUMAN_LABEL, 0))
    n_ai = int(counts.get(AI_LABEL, 0))
    if n_human == 0 or n_ai == 0:
        raise DeltxError(
            f"Cannot balance: human={n_human}, ai={n_ai} — need both classes present"
        )

    target = min(per_class, n_human, n_ai)
    if target < per_class:
        console.print(
            f"[yellow]Requested {per_class}/class but only {target} available "
            f"(human={n_human}, ai={n_ai}); using {target}.[/yellow]"
        )

    balanced = (
        unified.groupby("label", group_keys=False)
        .sample(n=target, random_state=seed)
        .sample(frac=1.0, random_state=seed)  # shuffle classes together
        .reset_index(drop=True)
    )
    return balanced


def report(unified: pd.DataFrame, balanced: pd.DataFrame) -> None:
    """Print class counts and the AI generator distribution for LOMO planning."""
    table = Table(title="Corpus assembly")
    table.add_column("Stage", style="bold")
    table.add_column("Human", justify="right")
    table.add_column("AI", justify="right")
    table.add_column("Total", justify="right")
    for name, frame in (("unified (filtered)", unified), ("balanced", balanced)):
        counts = frame["label"].value_counts()
        human = int(counts.get(HUMAN_LABEL, 0))
        ai = int(counts.get(AI_LABEL, 0))
        table.add_row(name, str(human), str(ai), str(human + ai))
    console.print(table)

    ai_rows = balanced.loc[balanced["label"] == AI_LABEL, "ai_model"]
    gen_counts = ai_rows.value_counts()
    gen_table = Table(
        title="AI generators in balanced set (pick --holdout-model from here)"
    )
    gen_table.add_column("ai_model (generator)")
    gen_table.add_column("count", justify="right")
    for name, count in gen_counts.items():
        gen_table.add_row(str(name), str(int(count)))
    console.print(gen_table)
    console.print(
        "[dim]For the leave-one-model-out test in Phase C, choose a generator "
        "with a healthy count (~500+ rows) so the held-out slice is meaningful."
        "[/dim]"
    )


def main(argv: list[str] | None = None) -> int:
    """Assemble and persist the balanced corpus. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    config = DeltxConfig()
    console.rule("[bold]Phase A — assemble balanced training corpus")

    per_class = args.per_class
    max_per_source = args.max_per_source
    if args.limit is not None:
        max_per_source = args.limit
        per_class = max(1, args.limit // 2)
        console.print(
            f"[yellow]--limit {args.limit}: max_per_source={max_per_source}, "
            f"per_class={per_class} (smoke test).[/yellow]"
        )

    try:
        manager = DatasetManager(config)
        if not args.skip_download:
            download_sources(manager, args.sources)
        unified = manager.load_and_unify(
            sources=args.sources, max_per_source=max_per_source
        )
        balanced = balance(unified, per_class, config.random_seed)
    except DeltxError as exc:
        console.print(f"[red]Corpus assembly failed:[/red] {exc}")
        return 1

    report(unified, balanced)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_parquet(args.output, index=False)
    console.print(
        f"[bold green]Balanced corpus saved →[/bold green] {args.output} "
        f"({len(balanced)} rows)"
    )
    console.print(
        "[bold]Next:[/bold] run Phase B (notebooks/extract_features_gpu.ipynb) on a "
        "GPU to turn this into data/processed/train_features.parquet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

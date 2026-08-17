"""Render static PNG charts from an extraction Parquet.

Point this at any results Parquet produced by ``deltx-extract`` and it writes a
small set of publication-quality figures next to it: the AI-confidence timeline
over the commit history, the distribution of scores, a per-author breakdown, a
repository-activity view, and a combined dashboard.

The charts follow a single-hue encoding throughout — commit AI-confidence is one
series (blue), magnitude is shown by position/length, and the 50%% decision line
is the one reserved status colour, always carrying a text label. No multi-hue
categorical palette is introduced, so the marks stay colour-blind safe by
construction.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - headless backend must precede pyplot import

import click  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from deltx.common.constants import EXTRACTION_COLUMNS  # noqa: E402
from deltx.common.exceptions import DeltxError, ExtractionError  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# --- Palette (reference data-viz instance, light surface) -----------------
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_BLUE = "#2a78d6"  # categorical slot 1 / series
_BLUE_DARK = "#184f95"  # sequential 600, for the trend line
_BLUE_MID = "#3987e5"  # sequential 400, for the secondary activity panel
_CRITICAL = "#d03b3b"  # reserved status colour for the decision threshold

_THRESHOLD = 50.0

_RC: dict[str, object] = {
    "figure.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "axes.edgecolor": _BASELINE,
    "axes.labelcolor": _INK2,
    "axes.titlecolor": _INK,
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "xtick.labelcolor": _INK2,
    "ytick.labelcolor": _INK2,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
}


@dataclass(frozen=True)
class ResultsSummary:
    """Headline statistics over an extraction result."""

    repo_label: str
    total_commits: int
    scored_commits: int
    unscored_commits: int
    mean_ai_pct: float
    median_ai_pct: float
    commits_over_threshold: int
    authors: int


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate an extraction Parquet.

    Args:
        path: Path to a Parquet written by ``deltx-extract``.

    Returns:
        The results frame, sorted oldest-first by ``commit_index``.

    Raises:
        ExtractionError: If the file is missing, unreadable, or not an
            extraction result.
    """
    if not path.exists():
        msg = f"results file not found: {path}"
        raise ExtractionError(msg)
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError, ImportError) as exc:
        msg = f"could not read results file {path}: {exc}"
        raise ExtractionError(msg) from exc

    missing = set(EXTRACTION_COLUMNS) - set(frame.columns)
    if missing:
        msg = f"{path} is not an extraction result; missing columns: {sorted(missing)}"
        raise ExtractionError(msg)
    return frame.sort_values("commit_index").reset_index(drop=True)


def _repo_label(frame: pd.DataFrame, fallback: str) -> str:
    """Derive a short repository name from the recorded URL."""
    if frame.empty or frame["repo_url"].isna().all():
        return fallback
    url = str(frame["repo_url"].iloc[0]).rstrip("/")
    name = url.rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name or fallback


def summarise(frame: pd.DataFrame, repo_label: str) -> ResultsSummary:
    """Compute headline statistics for logging and chart subtitles."""
    scored = frame["ai_confidence_pct"].dropna()
    return ResultsSummary(
        repo_label=repo_label,
        total_commits=len(frame),
        scored_commits=int(scored.size),
        unscored_commits=int(frame["ai_confidence_pct"].isna().sum()),
        mean_ai_pct=float(scored.mean()) if scored.size else float("nan"),
        median_ai_pct=float(scored.median()) if scored.size else float("nan"),
        commits_over_threshold=int((scored > _THRESHOLD).sum()),
        authors=int(frame["commit_author"].nunique()),
    )


def _style_axes(ax: Axes) -> None:
    """Apply the shared chrome: recessive grid, muted spines, no top/right."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)


def _empty_note(ax: Axes, message: str) -> None:
    """Render a centred note when there is nothing to plot."""
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=_MUTED,
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_timeline(ax: Axes, frame: pd.DataFrame, window: int) -> None:
    """AI-confidence per commit over history, with a rolling-mean trend."""
    _style_axes(ax)
    ax.set_title("AI-authorship confidence over commit history")
    ax.set_xlabel("commit index (oldest → newest)")
    ax.set_ylabel("ai_confidence_pct")
    ax.set_ylim(-6, 103)

    scored = frame.dropna(subset=["ai_confidence_pct"])
    if scored.empty:
        _empty_note(ax, "no scored commits")
        return

    ax.scatter(
        scored["commit_index"],
        scored["ai_confidence_pct"],
        s=24,
        color=_BLUE,
        alpha=0.55,
        edgecolor=_SURFACE,
        linewidth=0.5,
        label="commit",
        zorder=3,
    )

    effective = max(2, min(window, scored.shape[0]))
    rolling = (
        scored["ai_confidence_pct"]
        .rolling(effective, min_periods=max(2, effective // 3))
        .mean()
    )
    ax.plot(
        scored["commit_index"],
        rolling,
        color=_BLUE_DARK,
        linewidth=2,
        label=f"{effective}-commit rolling mean",
        zorder=4,
    )

    ax.axhline(
        _THRESHOLD, color=_CRITICAL, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2
    )
    x_max = float(frame["commit_index"].max())
    ax.text(
        x_max,
        _THRESHOLD + 1.5,
        "50% decision threshold",
        ha="right",
        va="bottom",
        color=_CRITICAL,
        fontsize=8.5,
    )

    unscored = frame[frame["ai_confidence_pct"].isna()]
    if not unscored.empty:
        ax.plot(
            unscored["commit_index"],
            [-3.0] * len(unscored),
            marker="|",
            linestyle="none",
            color=_MUTED,
            alpha=0.5,
            markersize=7,
            label=f"no Python changed ({len(unscored)})",
        )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        labelcolor=_INK2,
        ncols=1,
    )


def _plot_distribution(ax: Axes, frame: pd.DataFrame, bins: int) -> None:
    """Histogram of scored AI-confidence, with median and threshold marks."""
    _style_axes(ax)
    ax.set_title("Distribution of scored commits")
    ax.set_xlabel("ai_confidence_pct")
    ax.set_ylabel("commits")
    ax.set_xlim(0, 100)

    scored = frame["ai_confidence_pct"].dropna()
    if scored.empty:
        _empty_note(ax, "no scored commits")
        return

    ax.hist(
        scored,
        bins=bins,
        range=(0, 100),
        color=_BLUE,
        edgecolor=_SURFACE,
        linewidth=1.0,
        zorder=3,
    )
    median = float(scored.median())
    ax.axvline(median, color=_BLUE_DARK, linewidth=1.6, zorder=4)
    ax.text(
        median + 1.5,
        ax.get_ylim()[1] * 0.94,
        f"median {median:.0f}%",
        color=_BLUE_DARK,
        fontsize=8.5,
        va="top",
    )
    ax.axvline(
        _THRESHOLD, color=_CRITICAL, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4
    )
    ax.text(
        _THRESHOLD + 1.5,
        ax.get_ylim()[1] * 0.80,
        "50% threshold",
        color=_CRITICAL,
        fontsize=8.5,
        va="top",
    )


def _plot_by_author(ax: Axes, frame: pd.DataFrame, min_commits: int) -> None:
    """Mean AI-confidence per author, for authors with enough scored commits."""
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=_GRID, linewidth=0.6)
    ax.set_title(f"Mean AI-confidence by author (≥ {min_commits} scored commits)")
    ax.set_xlabel("mean ai_confidence_pct")
    ax.set_xlim(0, 100)

    scored = frame.dropna(subset=["ai_confidence_pct"])
    if scored.empty:
        _empty_note(ax, "no scored commits")
        return

    grouped = scored.groupby("commit_author")["ai_confidence_pct"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= min_commits].sort_values("mean")
    if grouped.empty:
        _empty_note(ax, f"no author has ≥ {min_commits} scored commits")
        return

    positions = range(len(grouped))
    ax.barh(list(positions), grouped["mean"], color=_BLUE, height=0.68, zorder=3)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(list(grouped.index), fontsize=9)
    for pos, (mean, size) in enumerate(
        zip(grouped["mean"], grouped["size"], strict=True)
    ):
        ax.text(
            mean + 1.2,
            pos,
            f"{mean:.0f}%  (n={int(size)})",
            va="center",
            ha="left",
            color=_INK2,
            fontsize=8.5,
        )
    ax.axvline(
        _THRESHOLD, color=_CRITICAL, linewidth=1.2, linestyle=(0, (5, 3)), zorder=4
    )


def _plot_activity(ax_files: Axes, ax_loc: Axes, frame: pd.DataFrame) -> None:
    """Two stacked panels: Python files changed and LOC scored, per commit."""
    for ax in (ax_files, ax_loc):
        _style_axes(ax)

    ax_files.set_title("Repository activity over history")
    ax_files.bar(
        frame["commit_index"],
        frame["files_changed_py"],
        color=_BLUE,
        width=1.0,
        zorder=3,
    )
    ax_files.set_ylabel("Python files\nchanged")

    ax_loc.bar(
        frame["commit_index"],
        frame["total_loc_scored"],
        color=_BLUE_MID,
        width=1.0,
        zorder=3,
    )
    ax_loc.set_ylabel("LOC scored")
    ax_loc.set_xlabel("commit index (oldest → newest)")


def _subtitle(summary: ResultsSummary) -> str:
    """One-line subtitle of headline statistics for a figure."""
    mean = "—" if summary.scored_commits == 0 else f"{summary.mean_ai_pct:.1f}%"
    median = "—" if summary.scored_commits == 0 else f"{summary.median_ai_pct:.1f}%"
    return (
        f"{summary.total_commits} commits · {summary.scored_commits} scored · "
        f"mean {mean} · median {median} · "
        f"{summary.commits_over_threshold} over 50% · {summary.authors} authors"
    )


def _finalize(fig: Figure, summary: ResultsSummary, out: Path, dpi: int) -> Path:
    """Add the shared subtitle, save the figure, and return its path."""
    fig.text(
        0.5, 0.965, _subtitle(summary), ha="center", va="top", color=_MUTED, fontsize=9
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def render_report(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    repo_label: str,
    rolling_window: int = 20,
    hist_bins: int = 20,
    min_author_commits: int = 3,
    dpi: int = 130,
) -> list[Path]:
    """Render the full set of figures for a results frame.

    Args:
        frame: A validated extraction frame (see :func:`load_results`).
        output_dir: Directory to write PNGs into; created if absent.
        repo_label: Short repository name used in titles and filenames.
        rolling_window: Commit window for the timeline trend line.
        hist_bins: Number of histogram bins for the distribution.
        min_author_commits: Minimum scored commits for an author to be charted.
        dpi: Output resolution.

    Returns:
        The paths of every PNG written, in render order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(frame, repo_label)
    written: list[Path] = []

    with plt.rc_context(_RC):  # type: ignore[arg-type, unused-ignore]
        fig, ax = plt.subplots(figsize=(11, 5))
        _plot_timeline(ax, frame, rolling_window)
        written.append(
            _finalize(fig, summary, output_dir / f"{repo_label}_timeline.png", dpi)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_distribution(ax, frame, hist_bins)
        written.append(
            _finalize(fig, summary, output_dir / f"{repo_label}_distribution.png", dpi)
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        _plot_by_author(ax, frame, min_author_commits)
        written.append(
            _finalize(fig, summary, output_dir / f"{repo_label}_by_author.png", dpi)
        )

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(11, 5), sharex=True, height_ratios=[1, 1]
        )
        _plot_activity(ax_top, ax_bottom, frame)
        written.append(
            _finalize(fig, summary, output_dir / f"{repo_label}_activity.png", dpi)
        )

        written.append(
            _render_dashboard(
                frame,
                summary,
                rolling_window,
                hist_bins,
                min_author_commits,
                output_dir,
                dpi,
            )
        )
    return written


def _render_dashboard(
    frame: pd.DataFrame,
    summary: ResultsSummary,
    rolling_window: int,
    hist_bins: int,
    min_author_commits: int,
    output_dir: Path,
    dpi: int,
) -> Path:
    """Combine the three headline charts into one dashboard figure."""
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.32, wspace=0.22)
    _plot_timeline(fig.add_subplot(grid[0, :]), frame, rolling_window)
    _plot_distribution(fig.add_subplot(grid[1, 0]), frame, hist_bins)
    _plot_by_author(fig.add_subplot(grid[1, 1]), frame, min_author_commits)
    fig.suptitle(
        f"AI-authorship signal — {summary.repo_label}",
        fontsize=16,
        fontweight="bold",
        color=_INK,
        x=0.5,
        y=0.99,
    )
    return _finalize(
        fig, summary, output_dir / f"{summary.repo_label}_dashboard.png", dpi
    )


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Extraction Parquet to visualise (as written by deltx-extract).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for the PNGs. Defaults to a folder beside the input.",
)
@click.option(
    "--rolling-window",
    default=20,
    show_default=True,
    help="Commit window for the timeline trend line.",
)
@click.option(
    "--bins", default=20, show_default=True, help="Histogram bins for the distribution."
)
@click.option(
    "--min-author-commits",
    default=3,
    show_default=True,
    help="Minimum scored commits for an author to appear in the by-author chart.",
)
@click.option("--dpi", default=130, show_default=True, help="Output resolution.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    input_path: Path,
    output_dir: Path | None,
    rolling_window: int,
    bins: int,
    min_author_commits: int,
    dpi: int,
    verbose: bool,
) -> None:
    """Render static PNG charts from an extraction results Parquet."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    try:
        frame = load_results(input_path)
        label = _repo_label(frame, input_path.stem)
        target = (
            output_dir
            if output_dir is not None
            else input_path.parent / f"{label}_charts"
        )
        summary = summarise(frame, label)
        written = render_report(
            frame,
            target,
            repo_label=label,
            rolling_window=rolling_window,
            hist_bins=bins,
            min_author_commits=min_author_commits,
            dpi=dpi,
        )
    except DeltxError as exc:
        raise click.ClickException(str(exc)) from exc

    mean = "n/a" if summary.scored_commits == 0 else f"{summary.mean_ai_pct:.1f}%"
    median = "n/a" if summary.scored_commits == 0 else f"{summary.median_ai_pct:.1f}%"
    console.print(
        f"\n[bold]{label}[/bold]: {summary.total_commits} commits, "
        f"{summary.scored_commits} scored, mean {mean}, median {median}, "
        f"{summary.commits_over_threshold} over 50%, {summary.authors} authors\n"
        f"[green]Wrote {len(written)} charts to[/green] {target}"
    )
    for path in written:
        console.print(f"  - {path.name}")


if __name__ == "__main__":
    main()

"""Command-line interface for Stage 1 history extraction (``deltx-extract``).

Clones a repository, walks every commit on a branch oldest-first, scores the
Python files each commit added or modified through DroidDetect, and writes one
row per commit to a Parquet file. Safe to interrupt and resume.
"""

import logging
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

from deltx.common.config import DeltxConfig
from deltx.common.constants import CHECKPOINT_INTERVAL
from deltx.common.exceptions import DeltxError
from deltx.detection.inference import AIDetectionInference
from deltx.extraction.git_history import clone_repository
from deltx.extraction.models import CommitRow
from deltx.extraction.pipeline import CommitAiConfidenceExtractor
from deltx.extraction.storage import load_resume, write_parquet

logger = logging.getLogger(__name__)
console = Console(stderr=True)


def _configure_logging(log_file: Path, verbose: bool) -> None:
    """Send logs to both the console (rich) and a file beside the output.

    Args:
        log_file: Destination for the plain-text log.
        verbose: Emit ``DEBUG`` when set, else ``INFO``.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    # Close and replace any handlers a previous invocation installed, so a
    # reused log path is not held open by a stale FileHandler.
    for handler in root.handlers:
        handler.close()
    root.handlers = [
        RichHandler(console=console, rich_tracebacks=True, show_path=False),
        file_handler,
    ]


def _on_rm_error(func: Callable[[str], object], path: str, _exc: BaseException) -> None:
    """Clear the read-only bit Git sets on pack files, then retry removal.

    Windows refuses to delete read-only files, which Git's object store is full
    of, so a naive ``rmtree`` of a clone fails there.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove_tree(path: Path) -> None:
    """Best-effort recursive delete that tolerates Git's read-only objects."""
    shutil.rmtree(path, onexc=_on_rm_error)


def _summarise(rows: list[CommitRow], console_out: Console) -> None:
    """Print an end-of-run summary over the produced rows."""
    scored = [r.ai_confidence_pct for r in rows if not math.isnan(r.ai_confidence_pct)]
    mean = sum(scored) / len(scored) if scored else float("nan")
    high = sum(1 for v in scored if v > 50.0)
    console_out.print(
        f"\n[bold green]Done.[/bold green] {len(rows)} commits recorded, "
        f"{len(scored)} with scoreable Python.\n"
        f"Mean ai_confidence_pct (scored commits): "
        f"{mean:.2f}\n"
        f"Commits > 50% AI confidence: {high}"
    )


@click.command()
@click.option(
    "--repo-url", required=True, help="Git clone URL of the repository to analyse."
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination Parquet file.",
)
@click.option(
    "--branch",
    default=None,
    help="Branch to traverse. Defaults to the repository's primary branch.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    help="Torch device for inference. 'auto' uses CUDA when available.",
)
@click.option(
    "--resume",
    "resume_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Existing partial Parquet to continue from.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    repo_url: str,
    output: Path,
    branch: str | None,
    device: str,
    resume_path: Path | None,
    verbose: bool,
) -> None:
    """Extract per-commit AI-authorship confidence from a repository's history.

    Scoring is one sequence per forward pass by design: DroidDetect pools with an
    unmasked mean, so batching with padding would make each file's score depend
    on its neighbours. There is deliberately no batch-size knob.
    """
    _configure_logging(output.with_suffix(".log"), verbose)
    config = DeltxConfig(device=device)
    logger.info("repository: %s", repo_url)
    logger.info("device: %s", config.resolved_device)

    workdir = Path(tempfile.mkdtemp(prefix="deltx-extract-"))
    clone_dir = workdir / "repo"
    try:
        repository = clone_repository(repo_url, clone_dir)
        resolved = repository.resolve_branch(branch)
        commits = repository.iter_commits(resolved)
        logger.info("branch %s: %d commits found", resolved, len(commits))

        if resume_path is not None:
            rows, processed = load_resume(resume_path)
        else:
            rows, processed = [], set()

        detector_desc = "loading DroidDetect checkpoint (first run downloads ~569 MB)"
        logger.info(detector_desc)
        inference = AIDetectionInference.from_config(config)
        extractor = CommitAiConfidenceExtractor(inference, repository, repo_url)

        new_rows = 0
        for index, commit in enumerate(
            tqdm(commits, desc="scoring commits", unit="commit")
        ):
            if commit.commit_hash in processed:
                continue
            try:
                row = extractor.build_row(commit, index)
            except DeltxError as exc:
                logger.warning(
                    "commit %s failed (%s); recording an empty row",
                    commit.commit_hash[:8],
                    exc,
                )
                row = CommitRow.empty(
                    repo_url=repo_url,
                    commit_hash=commit.commit_hash,
                    commit_timestamp=commit.timestamp,
                    commit_author=commit.author,
                    commit_message=commit.message,
                    commit_index=index,
                )
            rows.append(row)
            new_rows += 1
            if new_rows % CHECKPOINT_INTERVAL == 0:
                write_parquet(rows, output)
                logger.debug("checkpointed %d rows to %s", len(rows), output)

        write_parquet(rows, output)
        logger.info("wrote %d rows to %s", len(rows), output)
        _summarise(rows, console)
    except DeltxError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        _remove_tree(workdir)


if __name__ == "__main__":
    main()

"""Command-line interface for the detection module (``deltx-detect``)."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DeltxError
from deltx.detection.inference import AIDetectionInference, skip_reason
from deltx.detection.models import FileAnalysisResult

console = Console()


def _configure_logging(verbose: bool) -> None:
    """Route logging through rich at the requested level."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _result_row(result: FileAnalysisResult) -> tuple[str, str, str, str]:
    """Render one file result as a table row."""
    if not result.is_scored:
        return (str(result.file_path), "—", "skipped", result.error_message or "")
    label = result.distribution.predicted_label.name if result.distribution else "—"
    return (
        str(result.file_path),
        f"{result.ai_confidence * 100:.2f}",
        label,
        f"{result.lines_of_code} LOC, {result.chunk_count} chunk(s)",
    )


@click.group()
@click.version_option(package_name="deltx")
def cli() -> None:
    """Detect AI-authored Python code with DroidDetect-Base-Binary."""


@cli.command()
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Python file to analyse.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def analyze(file_path: Path, verbose: bool) -> None:
    """Analyse a single file and print the result as JSON."""
    _configure_logging(verbose)
    try:
        detector = AIDetectionInference.from_config(DeltxConfig())
        source = file_path.read_text(encoding="utf-8", errors="replace")
        result = detector.analyze_file(source, file_path)
    except DeltxError as exc:
        raise click.ClickException(str(exc)) from exc

    payload = result.model_dump(mode="json")
    payload["ai_confidence_pct"] = round(result.ai_confidence * 100, 4)
    console.print_json(json.dumps(payload))


@cli.command("analyze-dir")
@click.option(
    "--dir",
    "directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to analyse recursively.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def analyze_dir(directory: Path, verbose: bool) -> None:
    """Analyse every Python file in a directory as if it were one commit."""
    _configure_logging(verbose)

    candidates = [p for p in sorted(directory.rglob("*.py")) if skip_reason(p) is None]
    if not candidates:
        raise click.ClickException(f"no analysable Python files under {directory}")

    try:
        detector = AIDetectionInference.from_config(DeltxConfig())
        files = {
            path: path.read_text(encoding="utf-8", errors="replace")
            for path in candidates
        }
        result = detector.analyze_commit(
            files=files,
            commit_hash=f"dir:{directory.name}",
            timestamp=datetime.now(UTC),
        )
    except DeltxError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title=f"AI authorship — {directory}")
    table.add_column("File", overflow="fold")
    table.add_column("AI %", justify="right")
    table.add_column("Class")
    table.add_column("Detail")
    for file_result in result.file_results:
        table.add_row(*_result_row(file_result))
    console.print(table)
    console.print(
        f"\n[bold]ai_confidence_pct[/bold] = {result.ai_confidence_pct:.2f} "
        f"({result.files_analyzed} analysed, {result.files_skipped} skipped)"
    )


if __name__ == "__main__":
    cli()

"""Command-line interface for the AI authorship detection module.

Two subcommands, both driven by :class:`AIDetectionInference`::

    poetry run python -m deltx.detection.cli analyze --file path/to/module.py
    poetry run python -m deltx.detection.cli analyze-dir --dir path/to/package

``analyze`` scores one file and prints the full :class:`FileAnalysisResult` as
JSON; ``analyze-dir`` scores every ``.py`` file beneath a directory as a single
synthetic commit and prints a summary table.

Both need the trained classifier (``config.classifier_path``) and the code
language model in the local cache. A missing classifier is reported as a clean
one-line error and a non-zero exit, not a traceback.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DeltxError
from deltx.detection.inference import AIDetectionInference
from deltx.detection.models import CommitAnalysisResult

logger = logging.getLogger(__name__)
_console = Console()


def _build_detector() -> AIDetectionInference:
    """Construct the detector from the default configuration/environment."""
    return AIDetectionInference.from_config(DeltxConfig())


def _detector_or_exit() -> AIDetectionInference:
    """Build the detector, or print a friendly error and exit non-zero."""
    try:
        return _build_detector()
    except DeltxError as exc:
        _console.print(f"[red]Cannot start detector:[/red] {exc}")
        raise SystemExit(1) from exc


@click.group()
def cli() -> None:
    """Deltx - AI authorship detection."""


@cli.command()
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Python file to analyze.",
)
def analyze(file_path: Path) -> None:
    """Analyze a single Python file and print the result as JSON."""
    detector = _detector_or_exit()
    source = file_path.read_text(encoding="utf-8", errors="replace")
    result = detector.analyze_file(source, file_path)
    _console.print_json(result.model_dump_json())


@cli.command(name="analyze-dir")
@click.option(
    "--dir",
    "directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory whose .py files will be analyzed.",
)
def analyze_dir(directory: Path) -> None:
    """Analyze every .py file under a directory and print a summary."""
    detector = _detector_or_exit()
    files = _read_python_files(directory)
    if not files:
        _console.print(f"[yellow]No Python files found under {directory}[/yellow]")
        return

    result = detector.analyze_commit(
        files=files,
        commit_hash=directory.name or "directory",
        timestamp=datetime.now(UTC),
    )
    _render_commit(result, directory)


def _read_python_files(directory: Path) -> dict[Path, str]:
    """Read every ``.py`` file under ``directory`` into a ``{path: source}`` map.

    Undecodable or unreadable files are skipped with a warning; the analyzer's own
    filename filter (setup.py, conftest.py, __pycache__) is applied downstream.
    """
    files: dict[Path, str] = {}
    for path in sorted(directory.rglob("*.py")):
        try:
            files[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
    return files


def _render_commit(result: CommitAnalysisResult, root: Path) -> None:
    """Print a per-file table plus a one-line commit summary."""
    table = Table(title=f"AI authorship — {root}")
    table.add_column("File")
    table.add_column("Parseable", justify="center")
    table.add_column("LOC", justify="right")
    table.add_column("AI %", justify="right")
    for file_result in result.file_results:
        try:
            shown = file_result.file_path.relative_to(root)
        except ValueError:
            shown = file_result.file_path
        table.add_row(
            str(shown),
            "yes" if file_result.is_parseable else "no",
            str(file_result.lines_of_code),
            f"{file_result.ai_confidence * 100:.1f}",
        )
    _console.print(table)
    _console.print(
        f"[bold]{result.total_files_analyzed}[/bold] files analyzed, "
        f"[bold]{result.total_files_skipped}[/bold] skipped - "
        f"commit ai_confidence = [bold]{result.ai_confidence_pct:.1f}%[/bold]"
    )


def main() -> None:
    """Console-script entry point."""
    cli()


if __name__ == "__main__":
    cli()

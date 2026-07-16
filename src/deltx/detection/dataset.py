"""Dataset construction pipeline for the AI authorship detection module.

Turns public human-vs-AI code corpora into a single deduplicated, Python-only
training table, then into the feature matrix the classifier consumes::

    download_*()  →  load_and_unify()  →  extract_features_dataset()
                                      →  prepare_train_test_split()

Source registry
===============

Every source below was verified against its live origin; the counts are what the
publishers actually ship, not what the design doc estimated.

===================  =========================================  ================
Source key           Origin                                     Python samples
===================  =========================================  ================
``aigcodeset``       HF ``basakdemirok/AIGCodeSet``             4,755 human
                                                                2,828 AI
``droidcollection``  HF ``project-droid/DroidCollection``       ~262k (train)
``codenet``          IBM Project CodeNet, ``Python800`` subset  human only
``gptsniffer``       GitHub ``MDEGroup/GPTSniffer``             **none** — Java
===================  =========================================  ================

Two of these deserve a warning.

**DroidCollection is not binary.** Its ``Label`` column takes four values:
``HUMAN_GENERATED``, ``MACHINE_GENERATED``, ``MACHINE_REFINED`` (human code an
LLM rewrote) and ``MACHINE_GENERATED_ADVERSARIAL`` (LLM output deliberately
styled to read as human). Only the first two map cleanly onto the ``label ∈
{0, 1}`` contract, so :attr:`DatasetManager.DROID_LABEL_MAP` keeps those and
drops the rest. Admitting the adversarial rows in particular would teach the
classifier that human style *is* AI style, which is the one lesson it must not
learn. Override the class attribute to change that policy.

**GPTSniffer contributes nothing.** Its replication package holds 28,174 Java
files and 26 Python files, all of which are the tool's own source code rather
than samples. Deltx is Python-only, so the source survives here as a
manual-placement loader (see :meth:`DatasetManager.download_gptsniffer`) and
yields zero rows unless data is supplied by hand.
"""

from __future__ import annotations

import logging
import re
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, cast
from urllib.parse import urlparse

import pandas as pd
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from sklearn.model_selection import train_test_split

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DatasetError
from deltx.detection.models import FeatureVector

if TYPE_CHECKING:
    # Imported for typing only: `deltx.detection.pipeline` pulls in torch and the
    # transformers stack, and dataset construction must stay importable (and the
    # unification path testable) without paying that cost.
    from deltx.detection.pipeline import FeatureExtractionPipeline

logger = logging.getLogger(__name__)

HUMAN_LABEL: Final = 0
AI_LABEL: Final = 1

#: The schema every loader must produce, in column order.
UNIFIED_COLUMNS: Final[tuple[str, ...]] = (
    "source_code",
    "label",
    "source_dataset",
    "ai_model",
    "language",
)

#: Dataset keys accepted by :meth:`DatasetManager.load_and_unify`.
SOURCE_NAMES: Final[tuple[str, ...]] = (
    "aigcodeset",
    "droidcollection",
    "codenet",
    "gptsniffer",
)

TARGET_LANGUAGE: Final = "python"

# Spellings of Python seen across the four corpora, normalised to TARGET_LANGUAGE.
_LANGUAGE_ALIASES: Final[dict[str, str]] = {"py": "python", "python3": "python"}

# Snippets below this length carry no authorship signal and only add label noise.
_MIN_TOKEN_COUNT: Final = 10

# A deliberately crude tokenizer. It exists only to size-gate samples before the
# expensive real extraction; `PythonSourceParser` does the tokenization that
# matters. Running the true tokenizer over a million rows to discard a handful of
# one-liners would cost far more than the filter saves.
_TOKEN_PATTERN: Final = re.compile(r"\w+|[^\w\s]")

_CHECKPOINT_INTERVAL: Final = 500
_MAX_DOWNLOAD_ATTEMPTS: Final = 3
_RETRY_BACKOFF_SECONDS: Final = 2.0
_DOWNLOAD_CHUNK_BYTES: Final = 1 << 20

# Marks rows whose feature extraction failed. Persisted in the checkpoint so a
# resumed run does not retry them, and stripped from the returned frame.
_EXTRACTED_COLUMN: Final = "features_extracted"

_AIGCODESET_REPO: Final = "basakdemirok/AIGCodeSet"
_AIGCODESET_FILES: Final[tuple[str, ...]] = (
    "data/human_selected_dataset.csv",
    "data/created_dataset_with_llms.csv",
)
# The third CSV in the repo restates the other two and appends ~1.5k embedding
# columns; loading it would double every sample.
_AIGCODESET_SKIP_PREFIX: Final = "all_data_with_ada_embeddings"

_DROID_REPO: Final = "project-droid/DroidCollection"
_DROID_ALLOW_PATTERNS: Final[tuple[str, ...]] = ("data/*.parquet",)
# Read only what the schema needs; the parquet files carry nine columns and the
# unread five include the bulky sampling/rewriting parameter blobs.
_DROID_COLUMNS: Final[list[str]] = ["Code", "Label", "Language", "Generator"]

_CODENET_URL: Final = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud"
    "/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz"
)
_CODENET_ARCHIVE: Final = "Project_CodeNet_Python800.tar.gz"

_GPTSNIFFER_HOMEPAGE: Final = "https://github.com/MDEGroup/GPTSniffer"
_GPTSNIFFER_AI_MODEL: Final = "chatgpt"
_GPTSNIFFER_INSTRUCTIONS: Final = f"""# GPTSniffer — manual placement required

The GPTSniffer replication package ({_GPTSNIFFER_HOMEPAGE}) ships **28,174 Java
files and no Python samples**; its 26 `.py` files are the tool's own source code.
Deltx analyses Python only, so nothing is downloaded automatically.

## Expected format

To contribute samples, place Python files in two subdirectories of this folder:

```
gptsniffer/
├── human/    # human-written .py    → label 0
└── ai/       # ChatGPT-generated .py → label 1, ai_model "{_GPTSNIFFER_AI_MODEL}"
```

Files may be nested to any depth; every `*.py` beneath each directory is read as
one sample (UTF-8; undecodable files are skipped).

## Loading

Once the files are in place:

```python
manager.load_from_directory("gptsniffer")
```

Without them, `load_and_unify(...)` logs a warning and skips this source.
"""


def _approx_token_count(source: object) -> int:
    """Cheap proxy token count used only by the minimum-length filter."""
    if not isinstance(source, str):
        return 0
    return len(_TOKEN_PATTERN.findall(source))


def _empty_unified_frame() -> pd.DataFrame:
    """An empty frame carrying the unified schema and correct label dtype."""
    return pd.DataFrame({name: [] for name in UNIFIED_COLUMNS}).astype(
        {"label": "int64"}
    )


def _read_source_file(path: Path) -> str | None:
    """Read a UTF-8 source file, returning ``None`` if it cannot be decoded."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.debug("Skipping unreadable file %s", path)
        return None


def _with_retries[T](operation: Callable[[], T], description: str) -> T:
    """Run ``operation``, retrying with exponential backoff, then failing loudly.

    Args:
        operation: A no-argument thunk performing the network call.
        description: Human-readable name used in log lines and the final error.

    Returns:
        Whatever ``operation`` returns on its first successful attempt.

    Raises:
        DatasetError: If every attempt failed; chained to the last exception.
    """
    last_error: Exception | None = None
    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - retried, then raised as DatasetError
            last_error = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s",
                description,
                attempt,
                _MAX_DOWNLOAD_ATTEMPTS,
                exc,
            )
            if attempt < _MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise DatasetError(
        f"{description} failed after {_MAX_DOWNLOAD_ATTEMPTS} attempts"
    ) from last_error


def _download_progress() -> Progress:
    """A rich progress bar sized for byte-oriented transfers."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )


def _sample_progress() -> Progress:
    """A rich progress bar sized for row-oriented work (current/total)."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )


def _download_file(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` with a progress bar.

    Downloads into a ``.part`` sibling and renames only after the transfer is
    verified complete, so ``destination`` is either absent or whole. A short read
    is *not* end-of-file: a dropped connection returns an empty chunk exactly as
    a finished body does, so the byte count is checked against ``Content-Length``
    before the rename. Without that check a half-downloaded archive is
    indistinguishable from a good one, and the caller's "skip if it exists" guard
    would cache the corruption forever.

    A truncated transfer raises, which lets :func:`_with_retries` try again.

    Raises:
        DatasetError: If the URL is not HTTPS, or the body was truncated.
    """
    if urlparse(url).scheme != "https":
        raise DatasetError(f"Refusing to download from a non-HTTPS URL: {url}")

    partial_path = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Bandit flags urlopen because a non-http scheme (file://, ftp://) could read
    # a local path; the scheme is pinned to https immediately above.
    with urllib.request.urlopen(url) as response:  # noqa: S310
        expected = int(response.headers.get("Content-Length") or 0)
        received = 0
        with _download_progress() as progress, partial_path.open("wb") as handle:
            task = progress.add_task(
                f"Downloading {destination.name}", total=expected or None
            )
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                received += len(chunk)
                progress.advance(task, len(chunk))

    if expected and received != expected:
        # Discard the fragment so a retry starts from a clean slate.
        partial_path.unlink(missing_ok=True)
        raise DatasetError(
            f"Truncated download of {url}: received {received} of {expected} bytes"
        )

    partial_path.replace(destination)


def _extract_tarball(archive: Path, destination: Path) -> None:
    """Extract a gzipped tarball, refusing members that escape ``destination``."""
    logger.info("Extracting %s → %s", archive.name, destination)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data" rejects absolute paths, parent traversal, and device
        # nodes; without it a hostile archive could write anywhere on disk.
        tar.extractall(path=destination, filter="data")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path``, atomically, choosing format by suffix.

    Raises:
        DatasetError: If the suffix is neither ``.csv`` nor ``.parquet``.
    """
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise DatasetError(
            f"Unsupported table format {path.suffix!r}; use '.csv' or '.parquet'"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename: a crash mid-write must not destroy the
    # checkpoint that the run is relying on to resume.
    temp_path = path.with_name(path.name + ".tmp")
    if suffix == ".csv":
        # Pin the terminator: pandas defaults to os.linesep for path targets, and
        # source_code fields contain embedded newlines that must round-trip.
        frame.to_csv(temp_path, index=False, lineterminator="\n")
    else:
        frame.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def _read_table(path: Path) -> pd.DataFrame:
    """Read a table written by :func:`_write_table`.

    Raises:
        DatasetError: If the suffix is neither ``.csv`` nor ``.parquet``.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise DatasetError(
        f"Unsupported table format {path.suffix!r}; use '.csv' or '.parquet'"
    )


class DatasetManager:
    """Downloads, filters, and preprocesses datasets for classifier training."""

    #: DroidCollection's four-class ``Label`` reduced to the binary contract.
    #: ``MACHINE_REFINED`` and ``MACHINE_GENERATED_ADVERSARIAL`` are absent by
    #: design and their rows are dropped: refined code has mixed authorship, and
    #: adversarial code is AI output crafted to look human. Override on a
    #: subclass (or on the class itself) to admit them.
    DROID_LABEL_MAP: ClassVar[dict[str, int]] = {
        "HUMAN_GENERATED": HUMAN_LABEL,
        "MACHINE_GENERATED": AI_LABEL,
    }

    def __init__(self, config: DeltxConfig, data_dir: Path = Path("data")) -> None:
        """Create the manager and ensure its working directories exist.

        Args:
            config: Global configuration; supplies ``random_seed`` for the
                subsampling and splitting paths.
            data_dir: Root beneath which ``raw/`` and ``processed/`` are created.
        """
        self.config = config
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        for directory in (self.raw_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- downloads ----------------------------------------------------------

    @staticmethod
    def _is_populated(directory: Path, pattern: str) -> bool:
        """True if ``directory`` holds at least one file matching ``pattern``."""
        return directory.is_dir() and next(directory.glob(pattern), None) is not None

    def download_aigcodeset(self) -> Path:
        """Download AIGCodeSet: 4,755 human + 2,828 AI Python samples.

        Fetched from the HuggingFace mirror ``basakdemirok/AIGCodeSet``. (The
        GitHub repository named in the design doc does not exist.) The AI half
        was generated by CodeLlama 34B, Codestral 22B and Gemini 1.5 Flash; the
        human half is drawn from CodeNet.

        Returns:
            The directory holding the downloaded CSVs. Re-downloading is skipped
            when it already contains data.
        """
        destination = self.raw_dir / "aigcodeset"
        if self._is_populated(destination, "**/*.csv"):
            logger.info("AIGCodeSet already present at %s; skipping", destination)
            return destination

        from huggingface_hub import hf_hub_download

        destination.mkdir(parents=True, exist_ok=True)
        for filename in _AIGCODESET_FILES:
            _with_retries(
                partial(
                    hf_hub_download,
                    repo_id=_AIGCODESET_REPO,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(destination),
                ),
                f"AIGCodeSet download of {filename}",
            )
        self._log_download_summary("aigcodeset", destination)
        return destination

    def download_droidcollection(self) -> Path:
        """Download the DroidCollection parquet shards (Python filtered on load).

        The largest source by far: ~1.06M rows across seven languages, of which
        roughly 262k are Python. Only the parquet data files are fetched, and
        only four of their nine columns are ever read.

        Returns:
            The directory holding the downloaded parquet shards. Re-downloading
            is skipped when it already contains data.
        """
        destination = self.raw_dir / "droidcollection"
        if self._is_populated(destination, "**/*.parquet"):
            logger.info("DroidCollection already present at %s; skipping", destination)
            return destination

        from huggingface_hub import snapshot_download

        destination.mkdir(parents=True, exist_ok=True)
        _with_retries(
            partial(
                snapshot_download,
                repo_id=_DROID_REPO,
                repo_type="dataset",
                allow_patterns=list(_DROID_ALLOW_PATTERNS),
                local_dir=str(destination),
            ),
            "DroidCollection snapshot download",
        )
        self._log_download_summary("droidcollection", destination)
        return destination

    def download_codenet_python(self) -> Path:
        """Download IBM Project CodeNet's ``Python800`` subset (human only).

        The full CodeNet archive is 7.8 GB across 55 languages; the ``Python800``
        benchmark subset is a 29 MiB tarball of Python submissions, which is all
        Deltx needs. Every sample is human-written, so all rows carry ``label=0``.

        Returns:
            The directory the tarball was extracted into. Re-downloading and
            re-extraction are skipped when it already contains ``.py`` files.
        """
        destination = self.raw_dir / "codenet"
        if self._is_populated(destination, "**/*.py"):
            logger.info("CodeNet already present at %s; skipping", destination)
            return destination

        destination.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / _CODENET_ARCHIVE
        if not archive.exists():
            _with_retries(
                partial(_download_file, _CODENET_URL, archive),
                "CodeNet Python800 download",
            )
        _extract_tarball(archive, destination)
        self._log_download_summary("codenet", destination)
        return destination

    def download_gptsniffer(self) -> Path:
        """Prepare the GPTSniffer directory. Downloads nothing — see below.

        GPTSniffer's replication package holds 28,174 Java files and 26 Python
        files, and those 26 are the tool's own source rather than samples. Since
        Deltx is Python-only the package would contribute zero rows, so instead
        of fetching ~30k Java files this writes a ``README.md`` describing the
        layout :meth:`load_from_directory` expects (``human/`` and ``ai/``
        subdirectories of ``.py`` files) for manual placement.

        Returns:
            The directory, containing placement instructions and any samples the
            caller has already supplied.
        """
        destination = self.raw_dir / "gptsniffer"
        destination.mkdir(parents=True, exist_ok=True)
        instructions = destination / "README.md"
        if not instructions.exists():
            instructions.write_text(_GPTSNIFFER_INSTRUCTIONS, encoding="utf-8")

        if self._is_populated(destination, "**/*.py"):
            self._log_download_summary("gptsniffer", destination)
        else:
            logger.warning(
                "GPTSniffer ships no Python samples (%s); place files by hand, see %s",
                _GPTSNIFFER_HOMEPAGE,
                instructions,
            )
        return destination

    def _log_download_summary(self, source: str, directory: Path) -> None:
        """Load a freshly downloaded source and log its size and class balance."""
        try:
            frame = self.load_from_directory(source, directory)
        except DatasetError as exc:
            logger.warning("Downloaded %s but could not summarise it: %s", source, exc)
            return
        self._log_statistics(frame, f"Downloaded {source}")

    # -- loading ------------------------------------------------------------

    def load_from_directory(
        self, source: str, path: Path | None = None
    ) -> pd.DataFrame:
        """Load one dataset from disk into the unified schema.

        Works on any directory whose contents match the source's native layout,
        so a manually placed dataset loads identically to a downloaded one.

        Args:
            source: One of :data:`SOURCE_NAMES`.
            path: Directory to read. Defaults to ``raw_dir / source``.

        Returns:
            A DataFrame with exactly the :data:`UNIFIED_COLUMNS`. Unfiltered:
            language and length filtering happen in :meth:`load_and_unify`.

        Raises:
            DatasetError: If ``source`` is unknown, the directory is missing, or
                a loader produced a frame that does not satisfy the schema.
        """
        if source not in SOURCE_NAMES:
            raise DatasetError(
                f"Unknown dataset source {source!r}; expected one of {SOURCE_NAMES}"
            )
        directory = self.raw_dir / source if path is None else path
        if not directory.is_dir():
            raise DatasetError(f"No data directory for {source!r} at {directory}")

        loaders: dict[str, Callable[[Path], pd.DataFrame]] = {
            "aigcodeset": self._load_aigcodeset,
            "droidcollection": self._load_droidcollection,
            "codenet": self._load_codenet,
            "gptsniffer": self._load_gptsniffer,
        }
        return self._coerce_schema(loaders[source](directory), source)

    @staticmethod
    def _coerce_schema(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        """Project onto :data:`UNIFIED_COLUMNS` and normalise dtypes.

        Raises:
            DatasetError: If the loader omitted a required column.
        """
        missing = [name for name in UNIFIED_COLUMNS if name not in frame.columns]
        if missing:
            raise DatasetError(
                f"Loader for {source!r} omitted column(s): {', '.join(missing)}"
            )

        projected = frame.loc[:, list(UNIFIED_COLUMNS)].copy()
        if projected.empty:
            return projected.astype({"label": "int64"})

        projected["label"] = projected["label"].astype("int64")
        projected["source_dataset"] = projected["source_dataset"].astype(str)
        projected["language"] = projected["language"].astype(str)
        # Normalise pandas' several flavours of missing (NaN, NaT, pd.NA) to None
        # so `ai_model is None` is a reliable "human sample" test downstream.
        projected["ai_model"] = projected["ai_model"].where(
            projected["ai_model"].notna(), other=None
        )
        return projected.reset_index(drop=True)

    def _load_aigcodeset(self, directory: Path) -> pd.DataFrame:
        """Load AIGCodeSet's two CSVs (``code``, ``label``, ``LLM`` columns)."""
        frames: list[pd.DataFrame] = []
        for csv_path in sorted(directory.rglob("*.csv")):
            if csv_path.name.startswith(_AIGCODESET_SKIP_PREFIX):
                continue
            table = pd.read_csv(csv_path)
            labels = table["label"].astype("int64")
            frames.append(
                pd.DataFrame(
                    {
                        "source_code": table["code"],
                        "label": labels,
                        "source_dataset": "aigcodeset",
                        # The human CSV records LLM="Human"; only AI rows name a model.
                        "ai_model": table["LLM"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .where(labels == AI_LABEL, other=None),
                        # Only the human CSV carries a language column; the AI one
                        # is Python by construction.
                        "language": (
                            table["language"]
                            if "language" in table.columns
                            else TARGET_LANGUAGE
                        ),
                    }
                )
            )
        if not frames:
            logger.warning("aigcodeset: no CSV files found under %s", directory)
            return _empty_unified_frame()
        return pd.concat(frames, ignore_index=True)

    def _load_droidcollection(self, directory: Path) -> pd.DataFrame:
        """Load DroidCollection parquet shards, keeping mapped labels only."""
        shards = sorted(directory.rglob("*.parquet"))
        if not shards:
            logger.warning("droidcollection: no parquet under %s", directory)
            return _empty_unified_frame()

        frames: list[pd.DataFrame] = []
        unmapped_total = 0
        for shard in shards:
            table = pd.read_parquet(shard, columns=_DROID_COLUMNS)
            language = table["Language"].astype(str).str.strip().str.lower()
            table = table[language == TARGET_LANGUAGE]
            if table.empty:
                continue

            raw_labels = table["Label"].astype(str).str.strip().str.upper()
            labels = raw_labels.map(self.DROID_LABEL_MAP)
            mapped = labels.notna()
            unmapped_total += int((~mapped).sum())
            table, labels = table[mapped], labels[mapped].astype("int64")
            if table.empty:
                continue

            frames.append(
                pd.DataFrame(
                    {
                        "source_code": table["Code"].to_numpy(),
                        "label": labels.to_numpy(),
                        "source_dataset": "droidcollection",
                        "ai_model": (
                            table["Generator"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .where(labels == AI_LABEL, other=None)
                            .to_numpy()
                        ),
                        "language": TARGET_LANGUAGE,
                    }
                )
            )

        if unmapped_total:
            logger.info(
                "droidcollection: dropped %d Python rows whose label is outside %s",
                unmapped_total,
                sorted(self.DROID_LABEL_MAP),
            )
        if not frames:
            return _empty_unified_frame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _load_codenet(directory: Path) -> pd.DataFrame:
        """Load CodeNet ``.py`` submissions; every sample is human-written."""
        records: list[dict[str, object]] = []
        for py_path in sorted(directory.rglob("*.py")):
            source = _read_source_file(py_path)
            if source is None:
                continue
            records.append(
                {
                    "source_code": source,
                    "label": HUMAN_LABEL,
                    "source_dataset": "codenet",
                    "ai_model": None,
                    "language": TARGET_LANGUAGE,
                }
            )
        if not records:
            logger.warning("codenet: no Python files found under %s", directory)
            return _empty_unified_frame()
        logger.info("codenet: read %d human-written Python files", len(records))
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _load_gptsniffer(directory: Path) -> pd.DataFrame:
        """Load manually placed GPTSniffer samples from ``human/`` and ``ai/``."""
        records: list[dict[str, object]] = []
        for subdirectory, label in (("human", HUMAN_LABEL), ("ai", AI_LABEL)):
            root = directory / subdirectory
            if not root.is_dir():
                continue
            for py_path in sorted(root.rglob("*.py")):
                source = _read_source_file(py_path)
                if source is None:
                    continue
                records.append(
                    {
                        "source_code": source,
                        "label": label,
                        "source_dataset": "gptsniffer",
                        "ai_model": _GPTSNIFFER_AI_MODEL if label == AI_LABEL else None,
                        "language": TARGET_LANGUAGE,
                    }
                )
        if not records:
            logger.warning(
                "gptsniffer: no Python samples under %s; expected 'human/' and 'ai/' "
                "subdirectories (the public package is Java-only)",
                directory,
            )
            return _empty_unified_frame()
        return pd.DataFrame.from_records(records)

    # -- unification --------------------------------------------------------

    def load_and_unify(
        self,
        sources: list[str] | None = None,
        *,
        max_per_source: int | None = None,
    ) -> pd.DataFrame:
        """Load the requested datasets into one filtered, deduplicated frame.

        Pipeline: load each source's native format → map to the common schema →
        keep Python only → drop samples under :data:`_MIN_TOKEN_COUNT` tokens →
        drop exact duplicate ``source_code``.

        A source whose directory is absent or empty is skipped with a warning
        rather than failing the run, so a partial download still yields a usable
        training set.

        Args:
            sources: Which datasets to include; defaults to all of
                :data:`SOURCE_NAMES`.
            max_per_source: Optional per-source cap, sampled with
                ``config.random_seed``. CodeNet and DroidCollection together
                offer ~500k Python samples against a 10k–20k target, and scoring
                each one against a 350M-parameter LM is the pipeline's dominant
                cost. Applied at load time, so the filters below may reduce a
                source below its cap.

        Returns:
            A DataFrame with exactly the :data:`UNIFIED_COLUMNS`, indexed 0..n-1.

        Raises:
            DatasetError: If a source name is unknown, or if no requested source
                yielded any samples.
        """
        selected = list(SOURCE_NAMES) if sources is None else list(sources)
        unknown = [name for name in selected if name not in SOURCE_NAMES]
        if unknown:
            raise DatasetError(
                f"Unknown dataset source(s): {', '.join(unknown)}; "
                f"expected a subset of {SOURCE_NAMES}"
            )

        frames: list[pd.DataFrame] = []
        for source in selected:
            directory = self.raw_dir / source
            if not directory.is_dir():
                logger.warning(
                    "Skipping %r: nothing at %s (run the matching download_* method)",
                    source,
                    directory,
                )
                continue
            frame = self.load_from_directory(source, directory)
            if frame.empty:
                logger.warning("Skipping %r: no samples found in %s", source, directory)
                continue
            if max_per_source is not None and len(frame) > max_per_source:
                logger.info(
                    "Subsampling %r: %d → %d rows (seed=%d)",
                    source,
                    len(frame),
                    max_per_source,
                    self.config.random_seed,
                )
                frame = frame.sample(
                    n=max_per_source, random_state=self.config.random_seed
                )
            frames.append(frame)

        if not frames:
            raise DatasetError(
                "No dataset sources available; download at least one before unifying"
            )

        unified = pd.concat(frames, ignore_index=True)
        logger.info(
            "Loaded %d raw samples from %d source(s)", len(unified), len(frames)
        )

        unified = self._filter_language(unified)
        unified = self._filter_short_samples(unified)
        # Conflicts must be found before deduplication collapses the very rows
        # that disagree with each other.
        unified = self._drop_label_conflicts(unified)
        unified = self._deduplicate(unified)
        unified = unified.reset_index(drop=True)

        self._log_statistics(unified, "Unified dataset")
        return unified

    @staticmethod
    def _filter_language(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalise the language column and keep Python rows only."""
        normalised = frame["language"].astype(str).str.strip().str.lower()
        normalised = normalised.map(lambda value: _LANGUAGE_ALIASES.get(value, value))
        filtered = frame.assign(language=normalised)
        filtered = filtered[filtered["language"] == TARGET_LANGUAGE]

        removed = len(frame) - len(filtered)
        if removed:
            logger.info("Language filter: removed %d non-Python samples", removed)
        return filtered

    @staticmethod
    def _filter_short_samples(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop samples with fewer than :data:`_MIN_TOKEN_COUNT` tokens."""
        counts = frame["source_code"].map(_approx_token_count)
        filtered = frame[counts >= _MIN_TOKEN_COUNT]

        removed = len(frame) - len(filtered)
        if removed:
            logger.info(
                "Length filter: removed %d samples under %d tokens",
                removed,
                _MIN_TOKEN_COUNT,
            )
        return filtered

    @staticmethod
    def _drop_label_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
        """Remove every copy of a ``source_code`` that carries more than one label.

        An identical string labelled both human and AI is not a duplicate, it is a
        contradiction: the corpora disagree about who wrote it. It happens when an
        LLM reproduces a human solution verbatim, which AIGCodeSet does 103 times
        within its own two halves.

        Keeping one copy would resolve the contradiction by whichever file the
        loader happened to read first — for AIGCodeSet, the AI-labelled copy, only
        because ``created_dataset_with_llms.csv`` sorts before
        ``human_selected_dataset.csv``. Letting alphabetical order assign ground
        truth is indefensible, and either choice trains the classifier on a string
        that demonstrably belongs to both classes. So both copies go.

        Runs before :meth:`_deduplicate`, which would otherwise collapse the
        disagreeing rows and hide the conflict.
        """
        if frame.empty:
            return frame

        # Distinct (code, label) pairs: a code appearing twice here holds two
        # different labels. Cheaper than a groupby over the full frame.
        pairs = frame.loc[:, ["source_code", "label"]].drop_duplicates()
        conflicted = pairs.loc[pairs["source_code"].duplicated(), "source_code"]
        if conflicted.empty:
            return frame

        conflicting_codes = set(conflicted)
        filtered = frame[~frame["source_code"].isin(conflicting_codes)]

        logger.warning(
            "Label conflict: dropped %d rows across %d code strings labelled both "
            "human and AI",
            len(frame) - len(filtered),
            len(conflicting_codes),
        )
        return filtered

    @staticmethod
    def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop exact-match duplicate ``source_code``, keeping the first seen.

        Order follows the caller's ``sources`` list, so an earlier-listed dataset
        wins a collision. This matters: AIGCodeSet's human half is drawn from
        CodeNet, and 451 of its 4,755 human rows are byte-identical to a
        ``Python800`` submission.

        Which copy survives is safe to decide by load order only because
        :meth:`_drop_label_conflicts` has already removed the groups that disagree
        about the label; every remaining collision is between rows of the same
        class, so the winner carries the same ``label`` either way.
        """
        deduplicated = frame.drop_duplicates(subset="source_code", keep="first")

        removed = len(frame) - len(deduplicated)
        if removed:
            logger.info("Deduplication: removed %d exact-duplicate samples", removed)
        return deduplicated

    @staticmethod
    def _log_statistics(frame: pd.DataFrame, stage: str) -> None:
        """Log sample count, class balance, and per-dataset counts for ``frame``."""
        if frame.empty:
            logger.warning("%s: 0 samples", stage)
            return

        total = len(frame)
        counts = frame["label"].value_counts()
        human = int(counts.get(HUMAN_LABEL, 0))
        ai = int(counts.get(AI_LABEL, 0))
        logger.info(
            "%s: %d samples — human %d (%.1f%%), AI %d (%.1f%%)",
            stage,
            total,
            human,
            100.0 * human / total,
            ai,
            100.0 * ai / total,
        )
        if "source_dataset" in frame.columns:
            for name, count in frame["source_dataset"].value_counts().items():
                logger.info("  %s: %d", name, count)

    # -- feature extraction -------------------------------------------------

    def extract_features_dataset(
        self,
        df: pd.DataFrame,
        pipeline: FeatureExtractionPipeline,
        output_path: Path | None = None,
        *,
        checkpoint_every: int = _CHECKPOINT_INTERVAL,
    ) -> pd.DataFrame:
        """Extract the 16-D feature vector for every row of ``df``.

        Extraction over a large corpus takes hours, so progress is checkpointed
        to ``output_path`` every ``checkpoint_every`` rows *and* on the way out of
        an interrupt. Re-invoking with the same ``output_path`` resumes at the
        first unprocessed row.

        Samples the pipeline rejects (unparseable, or a feature family raised)
        are recorded in the checkpoint with NaN features so a resumed run does
        not retry them, then dropped from the returned frame — a partially-zeroed
        vector would teach the classifier that a parse failure is an authorship
        signature.

        Args:
            df: A frame carrying at least ``source_code``.
            pipeline: The extractor; only ``extract_features_only`` is called.
            output_path: Checkpoint destination, ``.csv`` or ``.parquet``. When
                ``None`` nothing is written and no resume is possible.
            checkpoint_every: Rows between checkpoint writes.

        Returns:
            ``df`` with the 16 feature columns appended, minus the rows whose
            extraction failed.

        Raises:
            DatasetError: If ``source_code`` is absent, ``checkpoint_every`` is
                below 1, or ``output_path`` has an unsupported suffix.
        """
        if "source_code" not in df.columns:
            raise DatasetError("Cannot extract features: no 'source_code' column")
        if checkpoint_every < 1:
            raise DatasetError(
                f"checkpoint_every must be >= 1, got {checkpoint_every}"
            )

        feature_names = FeatureVector.feature_names()
        frame = df.reset_index(drop=True)
        total = len(frame)

        records = self._resume(output_path, frame, feature_names)
        if records:
            logger.info(
                "Resuming from %s: %d/%d rows already extracted",
                output_path,
                len(records),
                total,
            )

        try:
            with _sample_progress() as progress:
                task = progress.add_task(
                    "Extracting features", total=total, completed=len(records)
                )
                for position in range(len(records), total):
                    records.append(
                        self._extract_row(
                            pipeline, frame.iloc[position], position, feature_names
                        )
                    )
                    progress.advance(task)
                    if output_path is not None and len(records) % checkpoint_every == 0:
                        self._write_checkpoint(output_path, frame, records)
        finally:
            # Runs on success, on KeyboardInterrupt, and on a pipeline crash, so
            # the work done before the interrupt survives it.
            if output_path is not None and records:
                self._write_checkpoint(output_path, frame, records)

        return self._finalise(frame, records, feature_names)

    @staticmethod
    def _extract_row(
        pipeline: FeatureExtractionPipeline,
        row: pd.Series,
        position: int,
        feature_names: list[str],
    ) -> dict[str, float | bool]:
        """Extract one row's features, or a NaN record if the pipeline rejects it."""
        dataset = row.get("source_dataset", "sample")
        file_path = Path(f"{dataset}/sample_{position:06d}.py")
        vector = pipeline.extract_features_only(str(row["source_code"]), file_path)
        if vector is None:
            return {
                **dict.fromkeys(feature_names, float("nan")),
                _EXTRACTED_COLUMN: False,
            }
        values = vector.to_array().tolist()
        return {
            **dict(zip(feature_names, values, strict=True)),
            _EXTRACTED_COLUMN: True,
        }

    @staticmethod
    def _write_checkpoint(
        output_path: Path,
        frame: pd.DataFrame,
        records: Sequence[dict[str, float | bool]],
    ) -> None:
        """Persist the first ``len(records)`` rows of ``frame`` with their features."""
        completed = frame.iloc[: len(records)].reset_index(drop=True)
        features = pd.DataFrame.from_records(list(records))
        _write_table(pd.concat([completed, features], axis=1), output_path)
        logger.debug(
            "Checkpointed %d/%d rows → %s", len(records), len(frame), output_path
        )

    @staticmethod
    def _resume(
        output_path: Path | None,
        frame: pd.DataFrame,
        feature_names: list[str],
    ) -> list[dict[str, float | bool]]:
        """Recover completed feature records from a checkpoint, if it is usable.

        A checkpoint is only trusted when its ``source_code`` column is a prefix
        of ``frame``'s. Anything else — a different dataset, a reordered frame, a
        truncated write — is discarded rather than silently misaligning features
        with the rows they describe. Returns ``[]`` when there is nothing to
        resume from.
        """
        if output_path is None or not output_path.exists():
            return []

        try:
            checkpoint = _read_table(output_path)
        except (OSError, ValueError, DatasetError) as exc:
            logger.warning("Ignoring unreadable checkpoint %s: %s", output_path, exc)
            return []

        required = [*feature_names, _EXTRACTED_COLUMN, "source_code"]
        missing = [name for name in required if name not in checkpoint.columns]
        if missing:
            logger.warning(
                "Ignoring checkpoint %s: missing column(s) %s",
                output_path,
                ", ".join(missing),
            )
            return []

        if len(checkpoint) > len(frame):
            logger.warning(
                "Ignoring checkpoint %s: it holds %d rows but the dataset has %d",
                output_path,
                len(checkpoint),
                len(frame),
            )
            return []

        expected = frame["source_code"].iloc[: len(checkpoint)].tolist()
        if checkpoint["source_code"].tolist() != expected:
            logger.warning(
                "Ignoring checkpoint %s: its rows do not prefix the dataset",
                output_path,
            )
            return []

        columns = [*feature_names, _EXTRACTED_COLUMN]
        restored = checkpoint.loc[:, columns].copy()
        restored[_EXTRACTED_COLUMN] = restored[_EXTRACTED_COLUMN].astype(bool)
        return cast(
            "list[dict[str, float | bool]]", restored.to_dict(orient="records")
        )

    def _finalise(
        self,
        frame: pd.DataFrame,
        records: Sequence[dict[str, float | bool]],
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Join features onto ``frame`` and drop the rows extraction rejected."""
        if not records:
            empty = frame.copy()
            for name in feature_names:
                empty[name] = pd.Series(dtype="float64")
            self._log_statistics(empty, "Feature matrix")
            return empty

        features = pd.DataFrame.from_records(list(records))
        joined = pd.concat([frame, features], axis=1)

        extracted = joined[_EXTRACTED_COLUMN].astype(bool)
        failed = int((~extracted).sum())
        if failed:
            logger.warning(
                "Dropping %d/%d samples whose feature extraction failed",
                failed,
                len(joined),
            )
        result = (
            joined[extracted].drop(columns=[_EXTRACTED_COLUMN]).reset_index(drop=True)
        )
        self._log_statistics(result, "Feature matrix")
        return result

    # -- splitting ----------------------------------------------------------

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_by: str = "label",
        holdout_model: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split ``df`` into training and test sets.

        With ``holdout_model`` set, every sample from that generator is moved to
        the test set before splitting — the leave-one-model-out protocol, which
        measures whether the classifier generalises to an LLM it never saw. The
        remainder is then split by ``test_size``, stratified on ``stratify_by``.

        The model match is exact and case-insensitive, not a substring test:
        ``"llama"`` must not silently sweep in ``"codellama"``. Pass the exact
        ``ai_model`` value, which loaders record lower-cased.

        Args:
            df: The feature matrix to split.
            test_size: Fraction of the non-held-out pool assigned to test.
            stratify_by: Column to stratify on; stratification is skipped (with a
                warning) when a class is too rare to appear in both splits.
            holdout_model: An ``ai_model`` value to route entirely into test.

        Returns:
            ``(train_df, test_df)``, each re-indexed from zero.

        Raises:
            DatasetError: If ``df`` is empty, ``stratify_by`` is absent,
                ``holdout_model`` matches no sample, or the holdout consumed
                every row.
        """
        if df.empty:
            raise DatasetError("Cannot split an empty dataset")
        if stratify_by not in df.columns:
            raise DatasetError(
                f"Cannot stratify by {stratify_by!r}: column not present"
            )

        frame = df.reset_index(drop=True)
        holdout = frame.iloc[0:0]
        pool = frame

        if holdout_model is not None:
            if "ai_model" not in frame.columns:
                raise DatasetError("holdout_model requires an 'ai_model' column")
            models = frame["ai_model"].fillna("").astype(str).str.strip().str.lower()
            mask = models == holdout_model.strip().lower()
            if not mask.any():
                available = sorted({name for name in models.unique() if name})
                raise DatasetError(
                    f"holdout_model {holdout_model!r} matches no samples; "
                    f"available ai_model values: {available}"
                )
            holdout, pool = frame[mask], frame[~mask]
            logger.info(
                "Leave-one-model-out: holding out %d %r samples for test",
                len(holdout),
                holdout_model,
            )

        if pool.empty:
            raise DatasetError(
                f"Holding out {holdout_model!r} consumed every sample; nothing to split"
            )

        train, test = train_test_split(
            pool,
            test_size=test_size,
            stratify=self._stratify_column(pool, stratify_by, test_size),
            random_state=self.config.random_seed,
            shuffle=True,
        )
        train = train.reset_index(drop=True)
        test = (
            test.reset_index(drop=True)
            if holdout.empty
            else pd.concat([test, holdout], ignore_index=True)
        )

        self._log_statistics(train, "Train split")
        self._log_statistics(test, "Test split")
        return train, test

    @staticmethod
    def _stratify_column(
        pool: pd.DataFrame, stratify_by: str, test_size: float
    ) -> pd.Series | None:
        """Return the stratification column, or ``None`` when it cannot be used.

        ``train_test_split`` raises if any class has fewer than two members, or if
        the test split would be smaller than the number of classes. Both are real
        possibilities on a small or heavily held-out pool, and neither is worth
        failing the run over, so stratification is dropped with a warning.
        """
        column = pool[stratify_by]
        counts = column.value_counts()

        if len(counts) < 2:
            logger.warning(
                "Stratification disabled: %r holds a single class", stratify_by
            )
            return None
        if int(counts.min()) < 2:
            logger.warning(
                "Stratification disabled: class %r has fewer than 2 samples",
                counts.idxmin(),
            )
            return None
        if int(len(pool) * test_size) < len(counts):
            logger.warning(
                "Stratification disabled: test split too small for %d classes",
                len(counts),
            )
            return None
        return column


def available_sources() -> Iterable[str]:
    """The dataset keys :meth:`DatasetManager.load_and_unify` accepts."""
    return SOURCE_NAMES


__all__ = [
    "AI_LABEL",
    "HUMAN_LABEL",
    "SOURCE_NAMES",
    "TARGET_LANGUAGE",
    "UNIFIED_COLUMNS",
    "DatasetManager",
    "available_sources",
]

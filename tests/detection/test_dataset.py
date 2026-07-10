"""Tests for the dataset construction pipeline.

Nothing here touches the network or the language model. Each source is faked in
its *native* on-disk format — AIGCodeSet's two CSVs, DroidCollection's parquet
shards, CodeNet's tree of ``.py`` files, GPTSniffer's ``human/`` and ``ai/``
subdirectories — so the loaders are exercised against the same shapes they meet
in production. Feature extraction is driven by a stub standing in for
:class:`~deltx.detection.pipeline.FeatureExtractionPipeline`, which also keeps
this module free of the torch import chain.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import pytest

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DatasetError
from deltx.detection import dataset as dataset_module
from deltx.detection.dataset import (
    AI_LABEL,
    HUMAN_LABEL,
    SOURCE_NAMES,
    UNIFIED_COLUMNS,
    DatasetManager,
    available_sources,
)
from deltx.detection.models import FeatureVector

_FEATURE_NAMES = FeatureVector.feature_names()

# Every snippet must clear the 10-token minimum, or the length filter removes it
# and the test is measuring the wrong thing.
HUMAN_ONE = "def add(first, second):\n    return first + second\n"
HUMAN_TWO = "class Counter:\n    def __init__(self):\n        self.total = 0\n"
AI_ONE = "def multiply(a: int, b: int) -> int:\n    '''Multiply.'''\n    return a * b\n"
AI_TWO = "def divide(x: float, y: float) -> float:\n    return x / y if y else 0.0\n"
CODENET_ONE = (
    "import sys\n\nvalues = [int(line) for line in sys.stdin]\nprint(sum(values))\n"
)
CODENET_TWO = (
    "n = int(input())\ntotal = 0\n"
    "for index in range(n):\n    total += index\nprint(total)\n"
)
SNIFFER_HUMAN = (
    "def reverse(items):\n    result = []\n"
    "    for item in items:\n        result.insert(0, item)\n    return result\n"
)
SNIFFER_AI = "def reverse(items: list) -> list:\n    return list(reversed(items))\n"

JAVA_SOURCE = (
    "public class Main {\n"
    "    public static void main(String[] a) { System.out.println(1); }\n"
    "}\n"
)


def _vector(seed: float) -> FeatureVector:
    """A FeatureVector whose 16 values are distinct, so column order is testable."""
    return FeatureVector(
        **{name: seed + index for index, name in enumerate(_FEATURE_NAMES)}
    )


class _StubPipeline:
    """Stands in for FeatureExtractionPipeline: no model, recorded calls.

    Args:
        reject: Source strings for which extraction "fails" (returns ``None``).
        raise_after: Raise on the call following this many successful ones,
            simulating an interrupt mid-run.
    """

    def __init__(
        self,
        *,
        reject: tuple[str, ...] = (),
        raise_after: int | None = None,
    ) -> None:
        self.reject = reject
        self.raise_after = raise_after
        self.seen: list[str] = []
        self.paths: list[Path] = []

    def extract_features_only(
        self, source_code: str, file_path: Path
    ) -> FeatureVector | None:
        if self.raise_after is not None and len(self.seen) >= self.raise_after:
            raise RuntimeError("simulated interrupt")
        self.seen.append(source_code)
        self.paths.append(file_path)
        if source_code in self.reject:
            return None
        return _vector(float(len(self.seen)))


@pytest.fixture
def manager(config: DeltxConfig, tmp_path: Path) -> DatasetManager:
    """A DatasetManager rooted at an empty temporary data directory."""
    return DatasetManager(config, data_dir=tmp_path)


# -- raw-format fixture writers ----------------------------------------------


def _write_aigcodeset(
    raw_dir: Path, *, java_row: bool = False, conflict: str | None = None
) -> None:
    """Write AIGCodeSet's two CSVs in their real column layout.

    Args:
        java_row: Append a non-Python row to the human CSV.
        conflict: A source string to append to *both* CSVs, so it carries
            ``label=0`` and ``label=1`` at once — the real corpus does this 103
            times, where an LLM reproduced a human solution verbatim.
    """
    directory = raw_dir / "aigcodeset"
    directory.mkdir(parents=True, exist_ok=True)

    human_codes = [HUMAN_ONE, HUMAN_TWO]
    languages = ["Python", "Python"]
    if java_row:
        human_codes.append(JAVA_SOURCE)
        languages.append("Java")

    ai_codes = [AI_ONE, AI_TWO]
    ai_models = ["GEMINI", "CODESTRAL"]

    if conflict is not None:
        human_codes.append(conflict)
        languages.append("Python")
        ai_codes.append(conflict)
        ai_models.append("GEMINI")

    pd.DataFrame(
        {
            "submission_id": [f"s{i}" for i in range(len(human_codes))],
            "problem_id": [f"p{i}" for i in range(len(human_codes))],
            "language": languages,
            "status_in_folder": ["Accepted"] * len(human_codes),
            "code": human_codes,
            "label": [HUMAN_LABEL] * len(human_codes),
            "LLM": ["Human"] * len(human_codes),
        }
    ).to_csv(directory / "human_selected_dataset.csv", index=False)

    pd.DataFrame(
        {
            "problem_id": [f"p{i}" for i in range(len(ai_codes))],
            "submission_id": [f"a{i}" for i in range(len(ai_codes))],
            "LLM": ai_models,
            "status_in_folder": ["Accepted"] * len(ai_codes),
            "code": ai_codes,
            "label": [AI_LABEL] * len(ai_codes),
        }
    ).to_csv(directory / "created_dataset_with_llms.csv", index=False)


def _write_droidcollection(raw_dir: Path, *, java_row: bool = False) -> None:
    """Write a DroidCollection parquet shard with its real nine columns."""
    directory = raw_dir / "droidcollection" / "data"
    directory.mkdir(parents=True, exist_ok=True)

    rows = [
        (HUMAN_ONE + "# droid\n", "HUMAN_GENERATED", "Python", "human"),
        (AI_ONE + "# droid\n", "MACHINE_GENERATED", "Python", "GPT-4o-mini"),
        # Neither cleanly human nor cleanly AI: dropped by DROID_LABEL_MAP.
        (AI_TWO + "# refined\n", "MACHINE_REFINED", "Python", "Qwen/Qwen2.5-72B"),
        (
            HUMAN_TWO + "# adv\n",
            "MACHINE_GENERATED_ADVERSARIAL",
            "Python",
            "GPT-4o-mini",
        ),
    ]
    if java_row:
        rows.append((JAVA_SOURCE, "MACHINE_GENERATED", "Java", "GPT-4o-mini"))

    pd.DataFrame(
        {
            "Code": [row[0] for row in rows],
            "Label": [row[1] for row in rows],
            "Language": [row[2] for row in rows],
            "Generator": [row[3] for row in rows],
            "Generation_Mode": ["INSTRUCT"] * len(rows),
            "Source": ["TACO"] * len(rows),
            "Sampling_Params": ["{}"] * len(rows),
            "Rewriting_Params": ["{}"] * len(rows),
            "Model_Family": ["openai"] * len(rows),
        }
    ).to_parquet(directory / "train-00000-of-00001.parquet", index=False)


def _write_codenet(raw_dir: Path, *, sources: tuple[str, ...] | None = None) -> None:
    """Write CodeNet's problem/submission tree of .py files."""
    payload = sources if sources is not None else (CODENET_ONE, CODENET_TWO)
    for index, source in enumerate(payload):
        problem = raw_dir / "codenet" / f"p{index:05d}"
        problem.mkdir(parents=True, exist_ok=True)
        (problem / f"s{index}.py").write_text(source, encoding="utf-8")


def _write_gptsniffer(raw_dir: Path) -> None:
    """Write GPTSniffer's manual-placement layout."""
    directory = raw_dir / "gptsniffer"
    (directory / "human").mkdir(parents=True, exist_ok=True)
    (directory / "ai").mkdir(parents=True, exist_ok=True)
    (directory / "human" / "a.py").write_text(SNIFFER_HUMAN, encoding="utf-8")
    (directory / "ai" / "b.py").write_text(SNIFFER_AI, encoding="utf-8")


# -- 1. unification across all four native formats ---------------------------


def test_load_and_unify_maps_every_source_to_the_common_schema(
    manager: DatasetManager,
) -> None:
    """All four native formats collapse into one correctly-typed frame."""
    _write_aigcodeset(manager.raw_dir)
    _write_droidcollection(manager.raw_dir)
    _write_codenet(manager.raw_dir)
    _write_gptsniffer(manager.raw_dir)

    unified = manager.load_and_unify()

    assert tuple(unified.columns) == UNIFIED_COLUMNS
    assert unified["label"].dtype == "int64"
    assert set(unified["label"]) == {HUMAN_LABEL, AI_LABEL}
    assert (unified["language"] == "python").all()

    # 4 AIGCodeSet + 2 Droid (of 4; two labels are unmapped) + 2 CodeNet + 2 sniffer.
    counts = unified["source_dataset"].value_counts().to_dict()
    assert counts == {
        "aigcodeset": 4,
        "droidcollection": 2,
        "codenet": 2,
        "gptsniffer": 2,
    }

    # ai_model is populated for AI rows and None for human ones.
    human_rows = unified[unified["label"] == HUMAN_LABEL]
    ai_rows = unified[unified["label"] == AI_LABEL]
    assert human_rows["ai_model"].isna().all()
    assert ai_rows["ai_model"].notna().all()
    assert {"gemini", "codestral", "gpt-4o-mini", "chatgpt"} <= set(ai_rows["ai_model"])


def test_load_and_unify_drops_droid_labels_outside_the_binary_map(
    manager: DatasetManager,
) -> None:
    """MACHINE_REFINED and MACHINE_GENERATED_ADVERSARIAL never reach the trainer."""
    _write_droidcollection(manager.raw_dir)

    unified = manager.load_and_unify(["droidcollection"])

    assert len(unified) == 2
    assert sorted(unified["label"]) == [HUMAN_LABEL, AI_LABEL]
    # The adversarial row was human-styled AI output; admitting it would poison
    # the human class.
    assert not unified["source_code"].str.contains("# adv").any()
    assert not unified["source_code"].str.contains("# refined").any()


def test_unknown_source_is_rejected(manager: DatasetManager) -> None:
    with pytest.raises(DatasetError, match="Unknown dataset source"):
        manager.load_and_unify(["not_a_dataset"])


def test_unify_with_no_available_sources_raises(manager: DatasetManager) -> None:
    with pytest.raises(DatasetError, match="No dataset sources available"):
        manager.load_and_unify(["codenet"])


# -- 2. language filtering ---------------------------------------------------


def test_non_python_samples_are_filtered_out(manager: DatasetManager) -> None:
    """A Java sample in either source is removed; only Python survives."""
    _write_aigcodeset(manager.raw_dir, java_row=True)
    _write_droidcollection(manager.raw_dir, java_row=True)

    unified = manager.load_and_unify(["aigcodeset", "droidcollection"])

    assert (unified["language"] == "python").all()
    assert not unified["source_code"].str.contains("public class Main").any()
    # 4 Python AIGCodeSet rows survive; the 5th (Java) does not.
    assert (unified["source_dataset"] == "aigcodeset").sum() == 4


# -- 3. deduplication --------------------------------------------------------


def test_exact_duplicate_source_code_is_removed(manager: DatasetManager) -> None:
    """The same file in two corpora yields one row, kept from the earlier source."""
    _write_aigcodeset(manager.raw_dir)
    # CodeNet re-serves a file that AIGCodeSet already carries. This is not
    # contrived: AIGCodeSet's human half is drawn from CodeNet.
    _write_codenet(manager.raw_dir, sources=(HUMAN_ONE, CODENET_ONE))

    unified = manager.load_and_unify(["aigcodeset", "codenet"])

    matches = unified[unified["source_code"] == HUMAN_ONE]
    assert len(matches) == 1
    assert matches.iloc[0]["source_dataset"] == "aigcodeset"
    assert not unified["source_code"].duplicated().any()
    assert len(unified) == 5  # 4 aigcodeset + 1 unique codenet


def test_samples_below_the_token_minimum_are_removed(manager: DatasetManager) -> None:
    """A one-liner carries no authorship signal and is dropped as noise."""
    _write_codenet(manager.raw_dir, sources=("x=1\n", CODENET_ONE))

    unified = manager.load_and_unify(["codenet"])

    assert len(unified) == 1
    assert unified.iloc[0]["source_code"] == CODENET_ONE


# -- label conflicts ---------------------------------------------------------


def test_label_conflicting_duplicates_are_dropped_entirely(
    manager: DatasetManager,
) -> None:
    """A string labelled both human and AI is a contradiction; both copies go.

    Keeping one would let the loader's file-read order assign ground truth: the
    AI copy wins today only because ``created_dataset_with_llms.csv`` sorts before
    ``human_selected_dataset.csv``.
    """
    _write_aigcodeset(manager.raw_dir, conflict=CODENET_ONE)

    unified = manager.load_and_unify(["aigcodeset"])

    # Neither the human-labelled nor the AI-labelled copy survives.
    assert CODENET_ONE not in set(unified["source_code"])
    assert (unified["source_code"] == CODENET_ONE).sum() == 0

    # The four unambiguous rows are untouched.
    assert len(unified) == 4
    assert set(unified["source_code"]) == {HUMAN_ONE, HUMAN_TWO, AI_ONE, AI_TWO}
    assert sorted(unified["label"]) == [HUMAN_LABEL, HUMAN_LABEL, AI_LABEL, AI_LABEL]


def test_label_conflict_is_detected_across_different_sources(
    manager: DatasetManager,
) -> None:
    """CodeNet says human, AIGCodeSet says AI: the row is dropped, not arbitrated."""
    _write_aigcodeset(manager.raw_dir)
    # AI_ONE ships as AI-labelled in AIGCodeSet; hand CodeNet the same bytes,
    # where every sample is human-labelled by construction.
    _write_codenet(manager.raw_dir, sources=(AI_ONE, CODENET_ONE))

    unified = manager.load_and_unify(["aigcodeset", "codenet"])

    assert AI_ONE not in set(unified["source_code"])
    # Source order must not decide the outcome either.
    reversed_order = manager.load_and_unify(["codenet", "aigcodeset"])
    assert AI_ONE not in set(reversed_order["source_code"])
    assert len(unified) == len(reversed_order) == 4


def test_agreeing_duplicates_are_deduplicated_not_dropped(
    manager: DatasetManager,
) -> None:
    """Same code, same label is a duplicate — collapse it, do not discard it."""
    _write_aigcodeset(manager.raw_dir)
    _write_codenet(manager.raw_dir, sources=(HUMAN_ONE, CODENET_ONE))

    unified = manager.load_and_unify(["aigcodeset", "codenet"])

    matches = unified[unified["source_code"] == HUMAN_ONE]
    assert len(matches) == 1
    assert matches.iloc[0]["label"] == HUMAN_LABEL
    assert matches.iloc[0]["source_dataset"] == "aigcodeset"


def test_conflict_removal_precedes_deduplication(manager: DatasetManager) -> None:
    """Dedup first would collapse the disagreeing rows and hide the conflict."""
    frame = pd.DataFrame(
        {
            "source_code": ["shared", "shared", "unique"],
            "label": [HUMAN_LABEL, AI_LABEL, HUMAN_LABEL],
            "source_dataset": ["a", "b", "a"],
            "ai_model": [None, "gpt4", None],
            "language": ["python"] * 3,
        }
    )

    survivors = manager._deduplicate(manager._drop_label_conflicts(frame))

    assert survivors["source_code"].tolist() == ["unique"]


def test_label_conflict_leaves_agreeing_rows_untouched(
    manager: DatasetManager,
) -> None:
    """Only the conflicting code string is removed, not its whole source."""
    conflicted = pd.DataFrame(
        {
            "source_code": ["shared", "shared", "keep_me", "keep_me"],
            "label": [HUMAN_LABEL, AI_LABEL, AI_LABEL, AI_LABEL],
            "source_dataset": ["a", "b", "a", "b"],
            "ai_model": [None, "gpt4", "gpt4", "claude"],
            "language": ["python"] * 4,
        }
    )

    filtered = manager._drop_label_conflicts(conflicted)

    assert filtered["source_code"].tolist() == ["keep_me", "keep_me"]


def test_frame_without_conflicts_passes_through_unchanged(
    manager: DatasetManager,
) -> None:
    frame = _labelled_frame(human=3, ai=2)

    pd.testing.assert_frame_equal(manager._drop_label_conflicts(frame), frame)


def test_drop_label_conflicts_handles_an_empty_frame(
    manager: DatasetManager,
) -> None:
    empty = _labelled_frame(0, 0)

    assert manager._drop_label_conflicts(empty).empty


# -- 4. stratified train/test split ------------------------------------------


def _labelled_frame(
    human: int, ai: int, ai_models: list[str] | None = None
) -> pd.DataFrame:
    """A minimal unified-schema frame with the requested class counts."""
    models = ai_models if ai_models is not None else ["gpt4"] * ai
    return pd.DataFrame(
        {
            "source_code": [f"# sample {index}\n" for index in range(human + ai)],
            "label": [HUMAN_LABEL] * human + [AI_LABEL] * ai,
            "source_dataset": ["synthetic"] * (human + ai),
            "ai_model": [None] * human + models,
            "language": ["python"] * (human + ai),
        }
    )


def test_split_is_stratified_and_correctly_sized(manager: DatasetManager) -> None:
    """100 samples (60 human / 40 AI) at test_size=0.2 → 20 test, balance preserved."""
    frame = _labelled_frame(human=60, ai=40)

    train, test = manager.prepare_train_test_split(frame, test_size=0.2)

    assert len(test) == 20
    assert len(train) == 80
    assert len(train) + len(test) == len(frame)

    # Stratification preserves the 60/40 ratio in each split exactly.
    assert (test["label"] == HUMAN_LABEL).sum() == 12
    assert (test["label"] == AI_LABEL).sum() == 8
    assert (train["label"] == HUMAN_LABEL).sum() == 48
    assert (train["label"] == AI_LABEL).sum() == 32

    # No sample appears in both halves.
    assert set(train["source_code"]).isdisjoint(set(test["source_code"]))


def test_split_is_deterministic_under_the_configured_seed(
    manager: DatasetManager,
) -> None:
    frame = _labelled_frame(human=60, ai=40)

    first_train, first_test = manager.prepare_train_test_split(frame)
    second_train, second_test = manager.prepare_train_test_split(frame)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_of_empty_frame_raises(manager: DatasetManager) -> None:
    with pytest.raises(DatasetError, match="empty dataset"):
        manager.prepare_train_test_split(_labelled_frame(0, 0))


def test_split_falls_back_when_a_class_is_too_rare_to_stratify(
    manager: DatasetManager,
) -> None:
    """A single AI sample cannot appear in both splits; stratification is dropped."""
    frame = _labelled_frame(human=20, ai=1)

    train, test = manager.prepare_train_test_split(frame, test_size=0.2)

    assert len(train) + len(test) == 21


# -- 5. leave-one-model-out --------------------------------------------------


def test_holdout_model_routes_every_matching_sample_into_test(
    manager: DatasetManager,
) -> None:
    """All Claude samples land in test; none leak into train."""
    models = ["gpt4"] * 10 + ["codellama"] * 10 + ["claude"] * 10
    frame = _labelled_frame(human=30, ai=30, ai_models=models)

    train, test = manager.prepare_train_test_split(
        frame, test_size=0.2, holdout_model="claude"
    )

    assert (test["ai_model"] == "claude").sum() == 10
    assert (train["ai_model"] == "claude").sum() == 0
    assert "claude" not in set(train["ai_model"].dropna())

    # The remaining 50 rows split 40/10, and the holdout is appended to test.
    assert len(train) == 40
    assert len(test) == 20
    assert len(train) + len(test) == len(frame)

    # The other two generators are still represented in training.
    assert {"gpt4", "codellama"} <= set(train["ai_model"].dropna())


def test_holdout_model_is_matched_case_insensitively(manager: DatasetManager) -> None:
    frame = _labelled_frame(human=30, ai=30, ai_models=["gpt4"] * 20 + ["claude"] * 10)

    _, test = manager.prepare_train_test_split(frame, holdout_model="CLAUDE")

    assert (test["ai_model"] == "claude").sum() == 10


def test_holdout_model_does_not_substring_match(manager: DatasetManager) -> None:
    """Holding out 'llama' must not sweep in 'codellama'."""
    frame = _labelled_frame(human=30, ai=30, ai_models=["codellama"] * 30)

    with pytest.raises(DatasetError, match="matches no samples"):
        manager.prepare_train_test_split(frame, holdout_model="llama")


def test_unknown_holdout_model_raises_and_lists_the_alternatives(
    manager: DatasetManager,
) -> None:
    frame = _labelled_frame(human=30, ai=30, ai_models=["gpt4"] * 30)

    with pytest.raises(DatasetError, match="gpt4"):
        manager.prepare_train_test_split(frame, holdout_model="gemini")


# -- 6. feature extraction over a dataset ------------------------------------


def test_extract_features_dataset_adds_all_sixteen_columns(
    manager: DatasetManager,
) -> None:
    """Every feature column is appended, in F1–F16 order, with the right values."""
    frame = _labelled_frame(human=3, ai=2)
    pipeline = _StubPipeline()

    result = manager.extract_features_dataset(frame, pipeline)  # type: ignore[arg-type]

    assert len(result) == 5
    for name in _FEATURE_NAMES:
        assert name in result.columns
    # Original columns survive alongside the features.
    assert set(UNIFIED_COLUMNS) <= set(result.columns)
    # The helper flag is an implementation detail of the checkpoint, not output.
    assert "features_extracted" not in result.columns

    # Row i was scored with seed i+1, so f1 == i+1 and f16 == i+16.
    assert result["f1_mean_surprisal"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["f16_hapax_legomena_ratio"].tolist() == [16.0, 17.0, 18.0, 19.0, 20.0]

    assert pipeline.seen == frame["source_code"].tolist()


def test_extract_features_dataset_drops_rows_the_pipeline_rejects(
    manager: DatasetManager,
) -> None:
    """A sample the pipeline cannot score is discarded, not zero-filled."""
    frame = _labelled_frame(human=3, ai=0)
    rejected = frame.iloc[1]["source_code"]
    pipeline = _StubPipeline(reject=(rejected,))

    result = manager.extract_features_dataset(frame, pipeline)  # type: ignore[arg-type]

    assert len(result) == 2
    assert rejected not in set(result["source_code"])
    assert result[_FEATURE_NAMES].notna().all().all()


def test_extract_features_dataset_requires_source_code(
    manager: DatasetManager,
) -> None:
    with pytest.raises(DatasetError, match="no 'source_code' column"):
        manager.extract_features_dataset(
            pd.DataFrame({"label": [0]}),
            _StubPipeline(),  # type: ignore[arg-type]
        )


# -- 7. checkpointing and resume ---------------------------------------------


def test_extraction_checkpoints_and_resumes_from_where_it_stopped(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """Interrupted after 5 rows, a restart re-scores rows 6..12 and nothing earlier."""
    frame = _labelled_frame(human=7, ai=5)  # 12 rows
    checkpoint = tmp_path / "features.csv"

    # First pass: blows up on the 6th row, after one checkpoint write at row 5.
    interrupted = _StubPipeline(raise_after=5)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        manager.extract_features_dataset(
            frame,
            interrupted,  # type: ignore[arg-type]
            output_path=checkpoint,
            checkpoint_every=5,
        )

    assert interrupted.seen == frame["source_code"].tolist()[:5]
    assert checkpoint.exists()
    saved = pd.read_csv(checkpoint)
    assert len(saved) == 5

    # Second pass: resumes at row 6 (index 5) and never re-scores rows 1–5.
    resumed = _StubPipeline()
    result = manager.extract_features_dataset(
        frame,
        resumed,  # type: ignore[arg-type]
        output_path=checkpoint,
        checkpoint_every=5,
    )

    assert resumed.seen == frame["source_code"].tolist()[5:]
    assert len(resumed.seen) == 7

    # The completed frame carries all 12 rows with features intact.
    assert len(result) == 12
    assert result[_FEATURE_NAMES].notna().all().all()
    assert result["source_code"].tolist() == frame["source_code"].tolist()

    # The first five rows still hold the values the interrupted run computed.
    assert result["f1_mean_surprisal"].tolist()[:5] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_completed_checkpoint_short_circuits_extraction(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """Re-running against a finished checkpoint calls the pipeline zero times."""
    frame = _labelled_frame(human=3, ai=2)
    checkpoint = tmp_path / "features.csv"

    manager.extract_features_dataset(
        frame,
        _StubPipeline(),  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    second = _StubPipeline()
    result = manager.extract_features_dataset(
        frame,
        second,  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    assert second.seen == []
    assert len(result) == 5


def test_mismatched_checkpoint_is_discarded_not_misaligned(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """A checkpoint from a different dataset must not be grafted onto this one."""
    checkpoint = tmp_path / "features.csv"
    original = _labelled_frame(human=4, ai=0)
    manager.extract_features_dataset(
        original,
        _StubPipeline(),  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    # A frame whose rows differ from the checkpoint's from the very first row.
    different = original.assign(
        source_code=[f"# different {index}\n" for index in range(4)]
    )
    pipeline = _StubPipeline()
    result = manager.extract_features_dataset(
        different,
        pipeline,  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    # Everything was re-extracted rather than resumed.
    assert pipeline.seen == different["source_code"].tolist()
    assert len(result) == 4


def test_checkpoint_round_trips_source_code_containing_newlines(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """Embedded newlines must survive the CSV round-trip, or resume misaligns."""
    frame = _labelled_frame(human=2, ai=0).assign(
        source_code=[HUMAN_ONE, CODENET_TWO]
    )
    checkpoint = tmp_path / "features.csv"

    manager.extract_features_dataset(
        frame,
        _StubPipeline(),  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    restored = pd.read_csv(checkpoint)
    assert restored["source_code"].tolist() == [HUMAN_ONE, CODENET_TWO]


def test_rejected_rows_are_not_rescored_after_a_resume(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """A sample the pipeline rejected stays rejected; the resume skips it."""
    frame = _labelled_frame(human=4, ai=0)
    rejected = frame.iloc[0]["source_code"]
    checkpoint = tmp_path / "features.csv"

    first = _StubPipeline(reject=(rejected,), raise_after=2)
    with pytest.raises(RuntimeError):
        manager.extract_features_dataset(
            frame,
            first,  # type: ignore[arg-type]
            output_path=checkpoint,
            checkpoint_every=1,
        )

    second = _StubPipeline()
    result = manager.extract_features_dataset(
        frame,
        second,  # type: ignore[arg-type]
        output_path=checkpoint,
        checkpoint_every=1,
    )

    # Row 0 was rejected on the first pass and is not offered to the second.
    assert rejected not in second.seen
    assert len(result) == 3
    assert rejected not in set(result["source_code"])


def test_unsupported_checkpoint_format_raises(
    manager: DatasetManager, tmp_path: Path
) -> None:
    with pytest.raises(DatasetError, match="Unsupported table format"):
        manager.extract_features_dataset(
            _labelled_frame(human=1, ai=0),
            _StubPipeline(),  # type: ignore[arg-type]
            output_path=tmp_path / "features.txt",
        )


def test_invalid_checkpoint_interval_raises(manager: DatasetManager) -> None:
    with pytest.raises(DatasetError, match="checkpoint_every"):
        manager.extract_features_dataset(
            _labelled_frame(human=1, ai=0),
            _StubPipeline(),  # type: ignore[arg-type]
            checkpoint_every=0,
        )


# -- directories and manual placement ----------------------------------------


def test_manager_creates_its_data_directories(
    config: DeltxConfig, tmp_path: Path
) -> None:
    built = DatasetManager(config, data_dir=tmp_path / "fresh")

    assert built.raw_dir.is_dir()
    assert built.processed_dir.is_dir()


def test_download_gptsniffer_writes_manual_instructions(
    manager: DatasetManager,
) -> None:
    """The Java-only package downloads nothing but documents how to supply data."""
    destination = manager.download_gptsniffer()

    instructions = destination / "README.md"
    assert instructions.is_file()
    body = instructions.read_text(encoding="utf-8")
    assert "human/" in body
    assert "ai/" in body
    # No samples were fetched.
    assert next(destination.glob("**/*.py"), None) is None


def test_load_from_directory_reads_a_manually_placed_dataset(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """A hand-populated directory loads identically to a downloaded one."""
    elsewhere = tmp_path / "manual" / "gptsniffer"
    (elsewhere / "human").mkdir(parents=True)
    (elsewhere / "ai").mkdir(parents=True)
    (elsewhere / "human" / "h.py").write_text(SNIFFER_HUMAN, encoding="utf-8")
    (elsewhere / "ai" / "a.py").write_text(SNIFFER_AI, encoding="utf-8")

    frame = manager.load_from_directory("gptsniffer", elsewhere)

    assert tuple(frame.columns) == UNIFIED_COLUMNS
    assert sorted(frame["label"]) == [HUMAN_LABEL, AI_LABEL]
    assert frame.loc[frame["label"] == AI_LABEL, "ai_model"].iloc[0] == "chatgpt"
    assert frame.loc[frame["label"] == HUMAN_LABEL, "ai_model"].iloc[0] is None


def test_load_from_directory_rejects_an_unknown_source(
    manager: DatasetManager,
) -> None:
    with pytest.raises(DatasetError, match="Unknown dataset source"):
        manager.load_from_directory("nonsense")


def test_load_from_directory_rejects_a_missing_directory(
    manager: DatasetManager,
) -> None:
    with pytest.raises(DatasetError, match="No data directory"):
        manager.load_from_directory("codenet")


def test_max_per_source_caps_each_dataset(manager: DatasetManager) -> None:
    """Subsampling is seeded, so a capped load is reproducible."""
    _write_codenet(
        manager.raw_dir,
        sources=(CODENET_ONE, CODENET_TWO, HUMAN_ONE, HUMAN_TWO, AI_ONE),
    )

    first = manager.load_and_unify(["codenet"], max_per_source=3)
    second = manager.load_and_unify(["codenet"], max_per_source=3)

    assert len(first) == 3
    assert first["source_code"].tolist() == second["source_code"].tolist()


def test_available_sources_matches_the_registry() -> None:
    assert tuple(available_sources()) == SOURCE_NAMES


def test_empty_source_directory_is_skipped(manager: DatasetManager) -> None:
    """A downloaded-but-empty source is skipped, not fatal."""
    (manager.raw_dir / "codenet").mkdir(parents=True)
    (manager.raw_dir / "droidcollection").mkdir(parents=True)
    _write_aigcodeset(manager.raw_dir)

    unified = manager.load_and_unify(["codenet", "droidcollection", "aigcodeset"])

    assert set(unified["source_dataset"]) == {"aigcodeset"}


def test_undecodable_source_file_is_skipped(manager: DatasetManager) -> None:
    """A file that is not valid UTF-8 is dropped rather than crashing the load."""
    _write_codenet(manager.raw_dir, sources=(CODENET_ONE,))
    broken = manager.raw_dir / "codenet" / "p99999"
    broken.mkdir(parents=True)
    (broken / "bad.py").write_bytes(b"\xff\xfe\x00 not utf-8 \xff")

    unified = manager.load_and_unify(["codenet"])

    assert len(unified) == 1
    assert unified.iloc[0]["source_code"] == CODENET_ONE


def test_extract_features_on_an_empty_frame_returns_empty_with_columns(
    manager: DatasetManager,
) -> None:
    result = manager.extract_features_dataset(
        _labelled_frame(0, 0),
        _StubPipeline(),  # type: ignore[arg-type]
    )

    assert result.empty
    for name in _FEATURE_NAMES:
        assert name in result.columns


def test_split_without_stratification_when_only_one_class_present(
    manager: DatasetManager,
) -> None:
    frame = _labelled_frame(human=10, ai=0)

    train, test = manager.prepare_train_test_split(frame, test_size=0.2)

    assert len(train) == 8
    assert len(test) == 2


# -- retry and download plumbing ---------------------------------------------


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries back off exponentially; tests must not actually wait."""
    monkeypatch.setattr(dataset_module.time, "sleep", lambda _seconds: None)


def test_with_retries_returns_after_a_transient_failure() -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("connection reset")
        return "downloaded"

    assert dataset_module._with_retries(flaky, "flaky download") == "downloaded"
    assert len(attempts) == 3


def test_with_retries_raises_dataset_error_after_three_attempts() -> None:
    attempts: list[int] = []

    def always_fails() -> str:
        attempts.append(1)
        raise OSError("no route to host")

    with pytest.raises(DatasetError, match="failed after 3 attempts") as excinfo:
        dataset_module._with_retries(always_fails, "doomed download")

    assert len(attempts) == 3
    # The final transport error is chained, not swallowed.
    assert isinstance(excinfo.value.__cause__, OSError)


def test_download_file_refuses_non_https_urls(tmp_path: Path) -> None:
    """A file:// or ftp:// URL would let a config change read arbitrary paths."""
    with pytest.raises(DatasetError, match="non-HTTPS"):
        dataset_module._download_file(
            "file:///etc/passwd", tmp_path / "payload.tar.gz"
        )


class _FakeResponse:
    """A urlopen response that may hand back fewer bytes than it advertises."""

    def __init__(self, body: bytes, advertised_length: int) -> None:
        self._body = body
        self._offset = 0
        self.headers = {"Content-Length": str(advertised_length)}

    def read(self, size: int) -> bytes:
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, body: bytes, advertised_length: int
) -> None:
    monkeypatch.setattr(
        dataset_module.urllib.request,
        "urlopen",
        lambda _url: _FakeResponse(body, advertised_length),
    )


def test_download_file_writes_a_complete_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * 4096
    _patch_urlopen(monkeypatch, payload, len(payload))
    destination = tmp_path / "payload.bin"

    dataset_module._download_file("https://example.test/payload.bin", destination)

    assert destination.read_bytes() == payload
    assert not destination.with_name("payload.bin.part").exists()


def test_truncated_download_is_rejected_and_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped connection returns an empty chunk exactly as a finished body does.

    Without the Content-Length check the fragment would be renamed into place and
    every later run would skip the download, caching the corruption forever.
    """
    _patch_urlopen(monkeypatch, b"only the first bytes" * 4, advertised_length=9999)
    destination = tmp_path / "payload.tar.gz"

    with pytest.raises(DatasetError, match="Truncated download"):
        dataset_module._download_file(
            "https://example.test/payload.tar.gz", destination
        )

    assert not destination.exists()
    assert not destination.with_name("payload.tar.gz.part").exists()


def test_truncated_download_is_retried_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Because truncation raises, _with_retries gets a second chance at it."""
    payload = b"complete body"
    attempts: list[int] = []

    def flaky_urlopen(_url: str) -> _FakeResponse:
        attempts.append(1)
        if len(attempts) == 1:
            return _FakeResponse(payload[:4], len(payload))  # short read
        return _FakeResponse(payload, len(payload))

    monkeypatch.setattr(dataset_module.urllib.request, "urlopen", flaky_urlopen)
    destination = tmp_path / "payload.bin"
    url = "https://example.test/p.bin"

    dataset_module._with_retries(
        lambda: dataset_module._download_file(url, destination),
        "flaky transfer",
    )

    assert len(attempts) == 2
    assert destination.read_bytes() == payload


def test_download_without_content_length_header_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunked response advertises no length; the size check must not fire."""
    payload = b"streamed body"
    _patch_urlopen(monkeypatch, payload, advertised_length=0)
    destination = tmp_path / "payload.bin"

    dataset_module._download_file("https://example.test/payload.bin", destination)

    assert destination.read_bytes() == payload


def _make_codenet_tarball(archive: Path, staging: Path) -> None:
    """Build a tarball shaped like Project_CodeNet_Python800."""
    root = staging / "Project_CodeNet_Python800" / "p00001"
    root.mkdir(parents=True)
    (root / "s000.py").write_text(CODENET_ONE, encoding="utf-8")
    (root / "s001.py").write_text(CODENET_TWO, encoding="utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(
            staging / "Project_CodeNet_Python800",
            arcname="Project_CodeNet_Python800",
        )


def test_download_codenet_extracts_the_tarball(
    manager: DatasetManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive is fetched once, then extracted into raw/codenet."""
    calls: list[str] = []

    def fake_download(url: str, destination: Path) -> None:
        calls.append(url)
        _make_codenet_tarball(destination, tmp_path / "staging")

    monkeypatch.setattr(dataset_module, "_download_file", fake_download)

    destination = manager.download_codenet_python()

    assert len(calls) == 1
    assert calls[0].endswith("Project_CodeNet_Python800.tar.gz")
    extracted = sorted(p.name for p in destination.rglob("*.py"))
    assert extracted == ["s000.py", "s001.py"]

    frame = manager.load_from_directory("codenet", destination)
    assert len(frame) == 2
    assert (frame["label"] == HUMAN_LABEL).all()


def test_download_codenet_skips_when_data_already_present(
    manager: DatasetManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing extraction short-circuits the download entirely."""
    _write_codenet(manager.raw_dir)

    def explode(url: str, destination: Path) -> None:
        raise AssertionError("should not re-download")

    monkeypatch.setattr(dataset_module, "_download_file", explode)

    assert manager.download_codenet_python() == manager.raw_dir / "codenet"


def test_download_aigcodeset_fetches_both_csvs(
    manager: DatasetManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    requested: list[str] = []

    def fake_hub_download(
        *, repo_id: str, filename: str, repo_type: str, local_dir: str
    ) -> str:
        requested.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        # Mirror the real column layout of whichever CSV was asked for.
        if "human" in filename:
            table = pd.DataFrame(
                {
                    "language": ["Python"],
                    "code": [HUMAN_ONE],
                    "label": [HUMAN_LABEL],
                    "LLM": ["Human"],
                }
            )
        else:
            table = pd.DataFrame(
                {"code": [AI_ONE], "label": [AI_LABEL], "LLM": ["GEMINI"]}
            )
        table.to_csv(target, index=False)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)

    destination = manager.download_aigcodeset()

    assert requested == [
        "data/human_selected_dataset.csv",
        "data/created_dataset_with_llms.csv",
    ]
    frame = manager.load_from_directory("aigcodeset", destination)
    assert sorted(frame["label"]) == [HUMAN_LABEL, AI_LABEL]

    # A second call must not re-download.
    requested.clear()
    manager.download_aigcodeset()
    assert requested == []


def test_download_droidcollection_fetches_only_parquet_shards(
    manager: DatasetManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    captured: dict[str, object] = {}

    def fake_snapshot(
        *, repo_id: str, repo_type: str, allow_patterns: list[str], local_dir: str
    ) -> str:
        captured["repo_id"] = repo_id
        captured["allow_patterns"] = allow_patterns
        _write_droidcollection(Path(local_dir).parent)
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    destination = manager.download_droidcollection()

    assert captured["repo_id"] == "project-droid/DroidCollection"
    assert captured["allow_patterns"] == ["data/*.parquet"]

    frame = manager.load_from_directory("droidcollection", destination)
    assert len(frame) == 2  # the refined and adversarial rows are dropped


def test_download_gptsniffer_summarises_manually_placed_samples(
    manager: DatasetManager,
) -> None:
    _write_gptsniffer(manager.raw_dir)

    destination = manager.download_gptsniffer()

    frame = manager.load_from_directory("gptsniffer", destination)
    assert len(frame) == 2


# -- checkpoint table formats and corruption ---------------------------------


def test_parquet_checkpoints_round_trip_and_resume(
    manager: DatasetManager, tmp_path: Path
) -> None:
    frame = _labelled_frame(human=4, ai=2)
    checkpoint = tmp_path / "features.parquet"

    interrupted = _StubPipeline(raise_after=3)
    with pytest.raises(RuntimeError):
        manager.extract_features_dataset(
            frame,
            interrupted,  # type: ignore[arg-type]
            output_path=checkpoint,
            checkpoint_every=3,
        )

    assert len(pd.read_parquet(checkpoint)) == 3

    resumed = _StubPipeline()
    result = manager.extract_features_dataset(
        frame,
        resumed,  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    assert len(resumed.seen) == 3
    assert len(result) == 6


def test_unreadable_checkpoint_is_ignored_and_work_restarts(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """A truncated or foreign file must not be mistaken for progress."""
    checkpoint = tmp_path / "features.csv"
    checkpoint.write_text("not,a,valid\ncheckpoint,at,all\n", encoding="utf-8")

    frame = _labelled_frame(human=2, ai=0)
    pipeline = _StubPipeline()
    result = manager.extract_features_dataset(
        frame,
        pipeline,  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    assert pipeline.seen == frame["source_code"].tolist()
    assert len(result) == 2


def test_checkpoint_longer_than_the_dataset_is_ignored(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """Resuming a 6-row checkpoint against a 2-row frame must not truncate it."""
    checkpoint = tmp_path / "features.csv"
    manager.extract_features_dataset(
        _labelled_frame(human=6, ai=0),
        _StubPipeline(),  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    smaller = _labelled_frame(human=2, ai=0)
    pipeline = _StubPipeline()
    result = manager.extract_features_dataset(
        smaller,
        pipeline,  # type: ignore[arg-type]
        output_path=checkpoint,
    )

    assert pipeline.seen == smaller["source_code"].tolist()
    assert len(result) == 2


def test_checkpoint_with_a_bad_suffix_is_ignored_on_read(
    manager: DatasetManager, tmp_path: Path
) -> None:
    """An existing file with an unsupported suffix is skipped, then rejected."""
    checkpoint = tmp_path / "features.txt"
    checkpoint.write_text("junk\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="Unsupported table format"):
        manager.extract_features_dataset(
            _labelled_frame(human=1, ai=0),
            _StubPipeline(),  # type: ignore[arg-type]
            output_path=checkpoint,
        )

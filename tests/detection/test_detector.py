"""Tests for chunking and source scoring.

Chunking carries a correctness requirement rather than a stylistic one: the
model scores indented, mid-scope fragments as machine-generated with high
confidence, so chunk boundaries must land on top-level statements.
"""

import pytest
import torch

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DetectionError
from deltx.detection.detector import (
    DroidDetector,
    ScoredSource,
    _top_level_segments,
    split_into_chunks,
)
from tests.conftest import FakeTokenizer

SOURCE = '''#!/usr/bin/env python
"""Module docstring."""

import os


def alpha():
    return os.getcwd()


class Beta:
    def method(self):
        return 1


def gamma():
    return 2
'''


class TestTopLevelSegments:
    def test_segments_reconstruct_the_source_exactly(self) -> None:
        """No line may be dropped or duplicated by segmentation."""
        segments = _top_level_segments(SOURCE)
        assert segments is not None
        assert "".join(segments) == SOURCE

    def test_every_segment_starts_at_column_zero(self) -> None:
        """This is the whole point: no segment may begin indented."""
        segments = _top_level_segments(SOURCE)
        assert segments is not None
        for segment in segments:
            first = segment.lstrip("\n").splitlines()[0]
            assert first == first.lstrip(), f"segment starts indented: {first!r}"

    def test_shebang_and_docstring_are_absorbed_into_the_first_segment(self) -> None:
        segments = _top_level_segments(SOURCE)
        assert segments is not None
        assert segments[0].startswith("#!/usr/bin/env python")

    def test_decorators_stay_with_their_function(self) -> None:
        source = "import x\n\n\n@decorator\ndef f():\n    pass\n"
        segments = _top_level_segments(source)
        assert segments is not None
        assert any(s.lstrip().startswith("@decorator") for s in segments)
        assert not any(s.lstrip().startswith("def f") for s in segments)

    def test_unparseable_source_returns_none(self) -> None:
        assert _top_level_segments("def broken(:\n") is None

    def test_empty_source_returns_none(self) -> None:
        assert _top_level_segments("") is None


class TestSplitIntoChunks:
    def test_short_source_is_a_single_chunk(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        chunks = split_into_chunks(SOURCE, fake_tokenizer, max_tokens=8192, stride=4096)
        assert len(chunks) == 1

    def test_empty_source_yields_no_chunks(self, fake_tokenizer: FakeTokenizer) -> None:
        assert split_into_chunks("", fake_tokenizer, 8192, 4096) == []

    def test_long_source_splits_on_top_level_boundaries(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """With a tight budget the file must split into several chunks."""
        source = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(40))
        chunks = split_into_chunks(source, fake_tokenizer, max_tokens=30, stride=15)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_the_budget(self, fake_tokenizer: FakeTokenizer) -> None:
        source = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(40))
        budget = 30 - fake_tokenizer.num_special_tokens_to_add()
        for chunk in split_into_chunks(source, fake_tokenizer, 30, 15):
            assert len(chunk) <= budget

    def test_oversized_single_block_falls_back_to_windows(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """One enormous function cannot be split cleanly; window it instead."""
        body = "\n".join(f"    value_{i} = {i}" for i in range(200))
        source = f"def huge():\n{body}\n"
        chunks = split_into_chunks(source, fake_tokenizer, max_tokens=40, stride=20)
        assert len(chunks) > 1

    def test_unparseable_long_source_falls_back_to_windows(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        source = "def broken(:\n" + "".join(f"    junk {i}\n" for i in range(200))
        chunks = split_into_chunks(source, fake_tokenizer, max_tokens=40, stride=20)
        assert len(chunks) > 1

    def test_stride_wider_than_budget_is_clamped(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """A stride past the budget would skip tokens; it must be clamped."""
        source = "def broken(:\n" + "".join(f"    junk {i}\n" for i in range(200))
        chunks = split_into_chunks(source, fake_tokenizer, max_tokens=40, stride=9999)
        assert all(len(c) <= 38 for c in chunks)


class StubModel:
    """Returns fixed logits, recording how many forward passes happened."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.calls = 0

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return torch.tensor([self.logits], dtype=torch.float)


@pytest.fixture
def detector(fake_tokenizer: FakeTokenizer) -> DroidDetector:
    """A detector whose model always predicts the same class."""
    config = DeltxConfig(device="cpu", max_sequence_length=8192, chunk_stride=4096)
    # Logits favouring MACHINE_GENERATED (index 1).
    return DroidDetector(StubModel([0.0, 10.0]), fake_tokenizer, config)  # type: ignore[arg-type]


class TestScoreSource:
    def test_returns_a_normalised_distribution(self, detector: DroidDetector) -> None:
        result = detector.score_source(SOURCE)
        assert isinstance(result, ScoredSource)
        values = result.distribution.model_dump().values()
        assert sum(values) == pytest.approx(1.0)

    def test_reports_token_and_chunk_counts(self, detector: DroidDetector) -> None:
        result = detector.score_source(SOURCE)
        assert result.token_count > 0
        assert result.chunk_count == 1

    def test_p_ai_reflects_the_prediction(self, detector: DroidDetector) -> None:
        result = detector.score_source(SOURCE)
        assert result.distribution.p_ai > 0.99

    def test_empty_source_raises(self, detector: DroidDetector) -> None:
        with pytest.raises(DetectionError, match="no tokens"):
            detector.score_source("   \n\n")

    def test_trivially_short_source_is_refused(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """Below ~10 tokens the score is erratic: a docstring-only __init__.py
        measures 0.406 where a single import line measures 0.981."""
        config = DeltxConfig(device="cpu", min_tokens_to_score=10)
        detector = DroidDetector(StubModel([0.0, 10.0]), fake_tokenizer, config)  # type: ignore[arg-type]
        with pytest.raises(DetectionError, match="token floor"):
            detector.score_source('"""A module."""\n')

    def test_floor_can_be_disabled(self, fake_tokenizer: FakeTokenizer) -> None:
        config = DeltxConfig(device="cpu", min_tokens_to_score=0)
        detector = DroidDetector(StubModel([0.0, 10.0]), fake_tokenizer, config)  # type: ignore[arg-type]
        assert detector.score_source('"""A module."""\n').token_count > 0

    def test_multi_chunk_source_runs_one_pass_per_chunk(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """Chunks are scored individually; nothing is padded into a batch."""
        config = DeltxConfig(device="cpu", max_sequence_length=30, chunk_stride=15)
        model = StubModel([0.0, 10.0])
        detector = DroidDetector(model, fake_tokenizer, config)  # type: ignore[arg-type]
        source = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(40))

        result = detector.score_source(source)
        assert result.chunk_count > 1
        assert model.calls == result.chunk_count

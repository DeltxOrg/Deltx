"""Tests for the perplexity feature extractor (F1–F6)."""

from __future__ import annotations

import math
import types

import pytest
import torch

from deltx.common.config import DeltxConfig
from deltx.detection.features.perplexity import PerplexityExtractor
from deltx.detection.models import SurprisalTrace

_FEATURE_KEYS = {
    "f1_mean_surprisal",
    "f2_surprisal_variance",
    "f3_sequence_perplexity",
    "f4_max_surprisal",
    "f5_low_surprisal_ratio",
    "f6_surprisal_slope",
}


@pytest.fixture
def extractor(config: DeltxConfig) -> PerplexityExtractor:
    """A PerplexityExtractor. Construction is lazy — no model is downloaded."""
    return PerplexityExtractor(config)


def _trace(values: list[float]) -> SurprisalTrace:
    return SurprisalTrace(
        surprisal_values=values,
        token_count=len(values) + 1,
        model_name="test-model",
    )


class _StubTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer."""

    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs: object) -> _StubTokenizer:
        return cls()

    def __call__(self, _text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


class _StubModel:
    """Minimal stand-in for a HuggingFace causal-LM (uniform logits)."""

    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs: object) -> _StubModel:
        return cls()

    def to(self, _device: str) -> _StubModel:
        return self

    def eval(self) -> _StubModel:
        return self

    def half(self) -> _StubModel:
        return self

    def float(self) -> _StubModel:
        return self

    def __call__(self, input_ids: torch.Tensor) -> types.SimpleNamespace:
        return types.SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 8))


class TestExtractFeatures:
    """Pure feature maths — runs without the language model."""

    def test_returns_exactly_six_named_keys(
        self, extractor: PerplexityExtractor
    ) -> None:
        features = extractor.extract_features(_trace([1.0, 2.0, 3.0]))
        assert set(features) == _FEATURE_KEYS
        assert len(features) == 6

    def test_known_trace_values(self, extractor: PerplexityExtractor) -> None:
        # Default low_surprisal_threshold is 2.0.
        assert extractor.config.low_surprisal_threshold == 2.0
        features = extractor.extract_features(_trace([2.0, 4.0, 1.0, 3.0, 5.0]))
        assert features["f1_mean_surprisal"] == pytest.approx(3.0)
        assert features["f2_surprisal_variance"] == pytest.approx(2.5)
        assert features["f4_max_surprisal"] == pytest.approx(5.0)
        # Only the single 1.0 value is below the 2.0 threshold → 1/5.
        assert features["f5_low_surprisal_ratio"] == pytest.approx(0.2)
        assert features["f6_surprisal_slope"] > 0  # increasing trend

    def test_empty_trace_returns_all_zeros(
        self, extractor: PerplexityExtractor
    ) -> None:
        features = extractor.extract_features(_trace([]))
        assert set(features) == _FEATURE_KEYS
        assert all(value == 0.0 for value in features.values())

    def test_single_value_trace_returns_all_zeros(
        self, extractor: PerplexityExtractor
    ) -> None:
        features = extractor.extract_features(_trace([3.14]))
        assert all(value == 0.0 for value in features.values())

    @pytest.mark.parametrize(
        "values",
        [
            [0.0, 0.0],  # perfectly certain model → perplexity exactly 1.0
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0, 3.0, 5.0],
            [15.6, 0.1],
        ],
    )
    def test_perplexity_at_least_one(
        self, extractor: PerplexityExtractor, values: list[float]
    ) -> None:
        features = extractor.extract_features(_trace(values))
        assert features["f3_sequence_perplexity"] >= 1.0


class TestComputeSurprisalTraceOffline:
    """Exercise the tensor pipeline with a stub model — no download required."""

    @staticmethod
    def _inject(
        extractor: PerplexityExtractor,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Wire a stub tokenizer/model that mimic the HuggingFace interfaces."""

        def fake_tokenizer(_text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {"input_ids": input_ids}

        def fake_model(_ids: torch.Tensor) -> types.SimpleNamespace:
            return types.SimpleNamespace(logits=logits)

        # Non-None attributes short-circuit lazy loading in _ensure_loaded.
        extractor._tokenizer = fake_tokenizer
        extractor._model = fake_model

    def test_uniform_logits_give_log2_vocab_bits(
        self, extractor: PerplexityExtractor
    ) -> None:
        vocab = 4
        input_ids = torch.tensor([[3, 1, 2, 0, 3]])  # 5 tokens
        logits = torch.zeros(1, 5, vocab)  # uniform ⇒ P = 1/vocab everywhere
        self._inject(extractor, input_ids, logits)

        trace = extractor.compute_surprisal_trace("ignored")

        # n tokens → n-1 scored positions.
        assert len(trace.surprisal_values) == 4
        assert trace.token_count == 5
        for surprisal in trace.surprisal_values:
            # −log₂(1/4) = 2 bits.
            assert surprisal == pytest.approx(math.log2(vocab))

    def test_surprisal_values_are_non_negative(
        self, extractor: PerplexityExtractor
    ) -> None:
        input_ids = torch.tensor([[5, 9, 2, 7, 1, 8]])
        logits = torch.randn(1, 6, 16)
        self._inject(extractor, input_ids, logits)

        trace = extractor.compute_surprisal_trace("ignored")

        assert len(trace.surprisal_values) == 5
        assert all(value >= 0.0 for value in trace.surprisal_values)

    def test_short_input_returns_empty_trace(
        self, extractor: PerplexityExtractor
    ) -> None:
        input_ids = torch.tensor([[1, 2]])  # < 3 tokens
        logits = torch.zeros(1, 2, 4)
        self._inject(extractor, input_ids, logits)

        trace = extractor.compute_surprisal_trace("x")

        assert trace.surprisal_values == []
        assert trace.token_count == 2

    def test_call_delegates_to_trace_then_features(
        self, extractor: PerplexityExtractor
    ) -> None:
        input_ids = torch.tensor([[3, 1, 2, 0, 3]])
        logits = torch.zeros(1, 5, 4)
        self._inject(extractor, input_ids, logits)

        features = extractor("ignored")

        assert set(features) == _FEATURE_KEYS
        # Uniform logits ⇒ constant 2-bit surprisal ⇒ perplexity 2**2 == 4.
        assert features["f3_sequence_perplexity"] == pytest.approx(4.0)


class TestModelLoadingAndFallback:
    """Lazy loading, device resolution, and CUDA-OOM fallback — via stubs."""

    def test_resolve_device_explicit(self) -> None:
        assert PerplexityExtractor._resolve_device("cpu") == "cpu"

    def test_resolve_device_auto(self) -> None:
        assert PerplexityExtractor._resolve_device("auto") in {"cpu", "cuda"}

    def test_lazy_load_on_first_inference(
        self, extractor: PerplexityExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "transformers.AutoTokenizer", _StubTokenizer, raising=False
        )
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM", _StubModel, raising=False
        )
        extractor.device = "cpu"

        assert extractor._model is None  # not loaded yet
        trace = extractor.compute_surprisal_trace("def f(): pass")

        assert isinstance(extractor._model, _StubModel)  # loaded on demand
        assert isinstance(extractor._tokenizer, _StubTokenizer)
        # 4 stub tokens → 3 scored positions, uniform logits → log2(8) bits each.
        assert len(trace.surprisal_values) == 3
        assert all(v == pytest.approx(math.log2(8)) for v in trace.surprisal_values)

    def test_cuda_oom_falls_back_to_cpu(
        self, extractor: PerplexityExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drive _surprisal_from_ids directly with a CPU tensor: on a CPU-only
        # torch build a real `.to("cuda")` cannot run, but the fallback logic
        # (re-run on CPU after an OOM) is exactly what we want to verify.
        extractor.device = "cuda"  # pretend we started on GPU
        extractor._tokenizer = _StubTokenizer()
        extractor._model = _StubModel()

        calls = {"n": 0}

        def flaky_forward(_input_ids: torch.Tensor) -> list[float]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("mock OOM")
            return [1.0, 2.0, 3.0]

        monkeypatch.setattr(extractor, "_forward_surprisal", flaky_forward)

        result = extractor._surprisal_from_ids(torch.tensor([[1, 2, 3, 4]]))

        assert calls["n"] == 2  # first OOMs, retry succeeds
        assert extractor.device == "cpu"  # fell back
        assert result == [1.0, 2.0, 3.0]


class TestComputeSurprisalTraceIntegration:
    """End-to-end check against the real CodeGen model (skipped if not cached)."""

    @pytest.mark.slow
    def test_produces_non_negative_values(
        self,
        require_model: None,  # gate fixture: skips if model not cached
        extractor: PerplexityExtractor,
    ) -> None:
        source = "def add(a, b):\n    return a + b\n"
        trace = extractor.compute_surprisal_trace(source)
        assert trace.token_count > 0
        assert all(value >= 0.0 for value in trace.surprisal_values)

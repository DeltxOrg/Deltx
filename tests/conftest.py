"""Shared fixtures.

No test in the default suite downloads the checkpoint. Anything requiring real
weights is marked ``slow`` and skipped unless explicitly selected.
"""

from pathlib import Path

import pytest

from deltx.common.config import DeltxConfig


@pytest.fixture
def config(tmp_path: Path) -> DeltxConfig:
    """A config pointed at a throwaway cache directory."""
    return DeltxConfig(model_cache_dir=tmp_path / "models", device="cpu")


class FakeTokenizer:
    """A whitespace tokenizer standing in for the real one.

    Only the surface ``split_into_chunks`` and ``DroidDetector`` actually use is
    implemented. Token counts are additive across line-aligned segments, which
    is the property the chunking tests depend on.
    """

    def __call__(
        self, text: str, add_special_tokens: bool = False, **kwargs: object
    ) -> dict[str, list[int]]:
        """Return one token id per whitespace-delimited word."""
        words = text.split()
        ids = [hash(word) % 50000 for word in words]
        if add_special_tokens:
            ids = self.build_inputs_with_special_tokens(ids)
        return {"input_ids": ids}

    def num_special_tokens_to_add(self) -> int:
        """The real tokenizer wraps input in ``[CLS]`` and ``[SEP]``."""
        return 2

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        """Wrap token ids in the sentinel pair."""
        return [50281, *token_ids, 50282]


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """A tokenizer stub requiring no download."""
    return FakeTokenizer()

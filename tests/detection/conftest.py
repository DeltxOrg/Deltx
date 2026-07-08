"""Detection-specific fixtures.

Provides ``require_model``: a gate that skips a test unless the CodeGen language
model is already present in ``config.model_cache_dir``. Tests exercising the real
model must depend on this fixture (and be marked ``@pytest.mark.slow``) so the
suite never triggers a multi-hundred-megabyte download on machines that have not
pre-fetched the model.
"""

from __future__ import annotations

import pytest

from deltx.common.config import DeltxConfig


def _model_is_cached(config: DeltxConfig) -> bool:
    """Return True if the model can be loaded from the local cache (offline)."""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(
            config.model_name,
            cache_dir=str(config.model_cache_dir),
            local_files_only=True,
        )
    except (OSError, ValueError):
        # transformers raises OSError/EnvironmentError when local_files_only is
        # set but the model is not present in the cache.
        return False
    return True


@pytest.fixture
def require_model(config: DeltxConfig) -> None:
    """Skip the requesting test unless the language model is cached locally."""
    if not _model_is_cached(config):
        pytest.skip(
            f"Language model {config.model_name} is not cached; skipping slow test"
        )

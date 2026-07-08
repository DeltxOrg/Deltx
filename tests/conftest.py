"""Shared pytest fixtures for Deltx tests."""

import pytest

from deltx.common.config import DeltxConfig


@pytest.fixture
def config() -> DeltxConfig:
    """Provide a default DeltxConfig instance."""
    return DeltxConfig()

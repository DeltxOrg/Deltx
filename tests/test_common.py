"""Tests for the common module."""

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import DeltxError


def test_config_instantiates() -> None:
    config = DeltxConfig()
    assert isinstance(config, DeltxConfig)


def test_deltx_error_is_exception() -> None:
    error = DeltxError("test")
    assert isinstance(error, Exception)
    assert str(error) == "test"


def test_version() -> None:
    from deltx import __version__

    assert __version__ == "0.1.0"

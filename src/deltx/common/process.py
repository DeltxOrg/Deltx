"""Subprocess boundary with bounded execution and redacted diagnostics."""

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from deltx.common.exceptions import DeltxError


def run_process(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 120,
    secrets: Sequence[str] = (),
    error_type: type[DeltxError] = DeltxError,
) -> bytes:
    """Run literal arguments without a shell; never expose supplied secrets."""
    try:
        # Callers provide command-specific literal argument vectors; shell=False
        # is the safety boundary for this shared executor.
        result = subprocess.run(
            list(args),  # noqa: S603
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            detail = ((exc.stdout or b"") + (exc.stderr or b"")).decode(
                "utf-8", errors="replace"
            )
        for secret in secrets:
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        message = f"{args[0]} failed (timeout={timeout}s): {detail}"
        if secrets:
            # The original exception may contain unredacted scanner output.
            raise error_type(message) from None
        raise error_type(message) from exc
    return result.stdout

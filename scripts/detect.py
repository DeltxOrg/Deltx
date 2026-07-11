#!/usr/bin/env python
"""Convenience wrapper: ``python scripts/detect.py ...`` → ``deltx.detection.cli``.

The canonical entry point is ``python -m deltx.detection.cli`` (or the
``deltx-detect`` console script); this file simply forwards to the same Click
group so the CLI is also runnable directly by path.
"""

from deltx.detection.cli import cli

if __name__ == "__main__":
    cli()

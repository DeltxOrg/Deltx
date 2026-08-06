#!/usr/bin/env python
"""Stage 1 history extraction entry point.

A thin shim over :func:`deltx.extraction.cli.main` so the tool can be run as a
plain script — ``python extract_ai_confidence.py --repo-url ... --output ...`` —
in addition to the installed ``deltx-extract`` console command. All logic lives
in the ``deltx.extraction`` package; this file only forwards to it.
"""

from deltx.extraction.cli import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Results-visualisation entry point.

A thin shim over :func:`deltx.extraction.visualize.main` so the charts can be
rendered as a plain script — ``python visualize_ai_confidence.py --input
results/repo_ai_confidence.parquet`` — in addition to the installed
``deltx-visualize`` console command. All logic lives in the package.
"""

from deltx.extraction.visualize import main

if __name__ == "__main__":
    main()

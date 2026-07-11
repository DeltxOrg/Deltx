# Deltx

Predictive Software Quality Analytics Platform. Deltx analyzes every commit in a Python repository's history, combining AI authorship detection, Squale-adapted quality scoring, PatchTST time-series forecasting, and SHAP explainability to predict software quality decay before it happens.

The **AI authorship detection module** (Stage 2) is implemented: it scores every commit with `ai_confidence_pct ∈ [0, 100]` — the probability that its code was LLM-generated — by scoring tokens against a Python-specific code language model (`Salesforce/codegen-350M-mono`), extracting a 16-feature vector (perplexity, stylometric, and distribution families), and classifying with XGBoost. File-level probabilities are LOC-weight-averaged to the commit level, and SHAP attribution explains each decision. See **[docs/detection/README.md](docs/detection/README.md)** for the full module documentation.

## Pipeline stages

| Stage | Module | Status |
|-------|--------|--------|
| 1. Data collection | `deltx.extraction` | planned |
| 2. AI authorship detection | `deltx.detection` | **implemented** |
| 3. Squale quality aggregation | `deltx.scoring` | planned |
| 4. PatchTST forecasting | `deltx.prediction` | planned |
| 5. SHAP explainability | `deltx.interpretation` | planned |

## Quick start

### Install

```bash
git clone https://github.com/your-org/deltx.git
cd deltx
poetry install
```

Requires Python 3.12. The first analysis run downloads the CodeGen-350M
language model (~700 MB) into `data/models/codegen`.

### Train (and validate) a detector

```bash
# Downloads a small AIGCodeSet sample, extracts features, trains XGBoost,
# prints metrics + SHAP report, and saves data/models/detector.joblib:
poetry run python scripts/validate_pipeline.py
```

For a full-size training run, use `DatasetManager` programmatically — see
[docs/detection/README.md](docs/detection/README.md#c-training-the-classifier-on-custom-data).

### Analyze code

```bash
# One file → JSON result
poetry run deltx-detect analyze --file path/to/module.py

# A whole directory → per-file table + commit-level ai_confidence_pct
poetry run deltx-detect analyze-dir --dir path/to/package
```

Or from Python:

```python
from deltx.common.config import DeltxConfig
from deltx.detection.inference import AIDetectionInference

detector = AIDetectionInference.from_config(DeltxConfig())
result = detector.analyze_commit(files, commit_hash, timestamp)
print(result.ai_confidence_pct)  # 0–100
```

### Run the tests

```bash
poetry run pytest              # unit + integration (offline, LM mocked)
poetry run ruff check src/
poetry run mypy src/
```

## Documentation

- [AI Detection Module](docs/detection/README.md) — architecture, feature
  taxonomy, usage, configuration, and API reference.

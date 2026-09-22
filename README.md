# Deltx

Predictive Software Quality Analytics Platform. Deltx analyzes every commit in a Python repository's history, combining AI authorship detection, Squale-adapted quality scoring, PatchTST time-series forecasting, and SHAP explainability to predict software quality decay before it happens.

Each commit is encoded as a 15-dimensional vector. Index `[4]` of that vector is `ai_confidence_pct ∈ [0, 100]` — the likelihood that the commit's code was LLM-generated — produced by the **AI authorship detection module** (Stage 2).

## Pipeline stages

| Stage | Module | Status |
|-------|--------|--------|
| 1. Data collection | `deltx.extraction` | **implemented** |
| 2. AI authorship detection | `deltx.detection` | **implemented** |
| 3. Squale quality aggregation | `deltx.scoring` | **implemented** |
| 4. PatchTST forecasting | `deltx.prediction` | planned |
| 5. SHAP explainability | `deltx.interpretation` | planned |

## AI authorship detection

Stage 2 is built on **[DroidDetect-Base-Binary](https://huggingface.co/project-droid/DroidDetect-Base-Binary)** — a published, Apache-2.0 detector for AI-generated code, fine-tuned from ModernBERT-base (149M params) on the DroidCollection corpus by Orel, Paul et al. ([*Droid: A Resource Suite for AI-Generated Code Detection*](https://arxiv.org/abs/2507.10583), EMNLP 2025).

Deltx trains and evaluates no detector of its own — the model arrives trained and benchmarked, and Python is one of the seven languages it covers. The suite publishes binary, ternary, and four-class heads; Deltx uses the binary one, which is both the most reliable (99.18 weighted F1, against 94.36 and 92.95) and the shape Stage 2 actually needs. The module wraps this checkpoint and does three things:

1. **Loads and runs it.** The checkpoint ships a custom head with no `auto_map`, so the architecture is defined locally in `detection/modeling.py` and the weights are loaded into it directly — `AutoModelForSequenceClassification` cannot read this repo. The export also carries the training-time triplet loss, which is filtered out before a strict load.
2. **Turns the prediction into one score.** DroidDetect-Base-Binary separates `HUMAN_GENERATED` from `MACHINE_GENERATED`, the latter covering generated, refined, and adversarial code alike. Deltx reports `ai_confidence_pct = 100 × (1 − P(HUMAN_GENERATED))`, and retains the full distribution for downstream stages.
3. **Aggregates file scores to the commit.** File-level probabilities are combined by LOC-weighted average. Files over ModernBERT's 8,192-token context are chunked and recombined by token-count weighting.

The score is the model's raw softmax mass rather than a calibrated probability, so read it as an ordinal confidence signal — which is what Stage 4 consumes it as.

## Quick start

### Install

```bash
git clone https://github.com/your-org/deltx.git
cd deltx
poetry install
```

Requires Python 3.12. The first run downloads the DroidDetect-Base-Binary
checkpoint (569 MB) into `data/models/droiddetect`.

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

print(result.ai_confidence_pct)  # 0–100, LOC-weighted across the commit
print(result.file_results[0].distribution)  # the retained per-file distribution
```

### Build a Python history dataset

```bash
docker compose -f docker/sonarqube/compose.yaml up -d sonarqube
# Complete SonarQube setup at http://localhost:19000 and set SONAR_TOKEN.
poetry run deltx dataset /path/to/repository --output dataset.csv
poetry run deltx dataset /path/to/repository --output all.csv --no-filter
```

The model CSV contains exactly 15 ordered numeric features; provenance is in
`dataset.metadata.csv`. Default selection skips semantically unchanged Python
checkpoints using AST comparison. `--no-filter` and `-no-filter` process every
checkpoint while keeping analysis Python-only. The user's working tree is untouched.

The pinned SonarQube/Scanner workflow waits for each analysis before collecting
current issues and metrics. Deltx then applies contextual issue weighting and
SQUALE-inspired nonlinear aggregation to four 0–100 quality scores. Parameters
are a reproducible research baseline and still require calibration.

See [the dataset and scoring guide](docs/scoring/README.md) for the architecture,
exact formulas, configuration, Docker setup, schema and limitations. Existing
`deltx-extract` AI-only Parquet extraction remains available.

### Run the tests

```bash
poetry run pytest              # unit + integration (offline, LM mocked)
poetry run pytest tests/scoring/ -v  # scoring module only
poetry run ruff check src/
poetry run mypy src/
```

Tests never download the checkpoint. Anything requiring real weights is marked
`@pytest.mark.slow` and deselected by `-m 'not slow'` in `addopts`; run those
with `poetry run pytest -m slow`.

## Scope

Python repositories only — no multi-language generalization.

Deltx inherits DroidDetect's known limitations rather than solving them: accuracy
degrades on generator families and coding domains outside its training distribution,
and adversarial humanization suppresses detection signals. `ai_confidence_pct` is
therefore a trend signal across a commit history, not a verdict on any single commit.

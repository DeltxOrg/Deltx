# Deltx — Predictive Software Quality Analytics Platform

## Project Identity

Deltx is a PhD research prototype (with product ambitions) that predicts software quality decay in Python repositories. It combines AI authorship detection, Squale-adapted quality scoring, PatchTST time-series forecasting, and SHAP explainability into a unified pipeline. Every commit in a repository's history is analyzed and encoded as a 15-dimensional vector that feeds the forecasting model.

**Scope constraint:** Python repositories only. No multi-language generalization.

## Monorepo Architecture

Five decoupled modules in a single repository:

```
src/deltx/
├── common/            # Shared data models, config, utilities
├── detection/         # Stage 2: AI Authorship Detection ← CURRENT FOCUS
│   ├── models.py      # Pydantic data models for detection
│   ├── modeling.py    # TLModel architecture + checkpoint loading
│   ├── detector.py    # DroidDetect wrapper: source → class probabilities
│   ├── inference.py   # File → commit-level inference pipeline
│   └── cli.py         # Command-line interface (deltx-detect)
├── extraction/        # Stage 1: Data Collection (future)
├── scoring/           # Stage 3: Squale Quality Aggregation (future)
├── prediction/        # Stage 4: PatchTST Forecasting (future)
└── interpretation/    # Stage 5: SHAP Explainability (future)
```

## Technology Stack

- **Python 3.12** with **Poetry** for dependency management
- **PyTorch** + **HuggingFace Transformers** for detector inference (ModernBERT encoder)
- **SHAP** for Stage 5 explainability over the 15-D commit vector
- **Pydantic v2** for all data models and validation
- **pandas** + **pyarrow** for tabular data; **huggingface-hub** for checkpoint downloads
- **pytest** for testing, **ruff** for linting/formatting, **mypy** for type checking
- **GitHub Actions** for CI/CD

> **pyarrow must be ≥ 24.0.** Earlier releases (16.x confirmed) ship DLLs that break
> `import torch` on Windows with `WinError 1114` on `c10.dll`. Because pandas
> auto-imports pyarrow when installed, `import pandas` alone is enough to trigger it.
> Likewise `huggingface-hub` is floored, not capped: capping it pins `transformers`
> to an old release.
>
> **transformers must be ≥ 4.48**, the first release with ModernBERT support.

## AI Detection Module — Complete Specification

### Purpose

Addresses the "Invisibility Gap": assigns each commit a probabilistic AI-authorship
score (`ai_confidence_pct`) quantifying the likelihood that its code was
LLM-generated. This score occupies index `[4]` of the 15-dimensional data vector as
an Evolutionary Driver.

Detection is performed by **DroidDetect-Base**, a published pre-trained
AI-generated-code detector, used directly as the Stage 2 detector. Deltx neither
trains nor re-evaluates a detector: the model is already trained, benchmarked, and
peer-reviewed by its authors, and Deltx consumes it as a fixed component. All
engineering effort goes into the wrapper — loading the checkpoint correctly, reading
its outputs correctly, and aggregating file-level predictions to the commit level.

### Detection Model

**Model:** `project-droid/DroidDetect-Base` (HuggingFace), Apache-2.0.
**Paper:** Orel, Paul et al., *Droid: A Resource Suite for AI-Generated Code
Detection*, EMNLP 2025 Main ([arXiv:2507.10583](https://arxiv.org/abs/2507.10583)).

| Property           | Value                                                          |
|--------------------|----------------------------------------------------------------|
| Backbone           | `answerdotai/ModernBERT-base` (encoder-only, 149M params)       |
| Head               | mean-pool → `Linear(768 → 128)` → ReLU → `Linear(128 → 4)`      |
| Task               | 4-class classification                                          |
| Context window     | 8,192 tokens (ModernBERT native) — most source files fit whole   |
| Training corpus    | Filtered `project-droid/DroidCollection` (7 languages incl. Python) |
| Training objective | `CrossEntropyLoss + 0.1 × BatchHardSoftMarginTripletLoss`        |

A `-Large` variant exists (ModernBERT-large, 396M). Base is the default: the
reported weighted-F1 gap is well under a point (see below) while Base is ~2.7×
smaller, and Stage 2 runs over every commit in a repository's history.

**In-domain weighted F1, as reported in the paper.** These are the authors'
published numbers and are the basis on which Deltx adopts the model; Deltx does not
reproduce them.

| Variant           | 2-class | 3-class | 4-class |
|-------------------|---------|---------|---------|
| DroidDetect-Base  | 99.18   | 94.36   | 92.95   |
| DroidDetect-Large | 99.25   | 95.17   | 94.30   |

The 2-class column is the operating point Deltx actually uses — see *Deriving
ai_confidence_pct* below.

### Checkpoint Loading Contract

The HF repo ships exactly: `config.json`, `pytorch_model.bin`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`.
There is **no modeling module and no `auto_map`**, and `config.json` is minimal:

```json
{"model_type": "custom_model", "architectures": ["Model"],
 "projection_dim": 128, "num_classes": 4}
```

Consequences, all load-bearing:

1. **`AutoModelForSequenceClassification.from_pretrained` cannot load this
   checkpoint.** `modeling.py` must define the `TLModel` architecture locally
   (mirroring the model card verbatim), instantiate `ModernBERT-base` as the
   `text_encoder`, then load `pytorch_model.bin` into it with `torch.load`.
2. **The tokenizer loads normally** via `AutoTokenizer.from_pretrained` — the
   ModernBERT tokenizer files are present and complete.
3. **Filter `additional_loss.*` keys when loading the state dict.** The training
   `TLModel.__init__` wraps the encoder in a `sentence-transformers`
   `BatchHardSoftMarginTripletLoss`, so the checkpoint may carry a duplicate copy
   of the encoder weights under that prefix. Inference omits the loss module
   entirely; load with explicit key filtering and assert that no *expected* key is
   missing, rather than passing a blanket `strict=False` that would silently
   tolerate an unloaded classifier head.
4. **Do not add `sentence-transformers` as a runtime dependency.** It is needed
   only to construct the training-time loss.

> **Padding contaminates the pooled embedding.** The model card's `forward` pools
> with `last_hidden_state.mean(dim=1)` — an unmasked mean over *every* position,
> including padding. A file's prediction therefore depends on what else is in its
> batch. Reproducing training-time behaviour means reproducing this pooling as
> written; do **not** "fix" it to a mask-aware mean, which would change the
> operating point the reported metrics describe. Instead avoid the ambiguity at
> inference: tokenize without padding and score one file per forward pass, or
> bucket files by token length. Any batching strategy must be verified to produce
> predictions identical to single-sample scoring before it is used.

### Label Semantics

The four classes are DroidCollection's `Label` values:

| Index | Label                           | Meaning                                        |
|-------|---------------------------------|------------------------------------------------|
| 0     | `HUMAN_GENERATED`               | Human-authored                                  |
| 1     | `MACHINE_GENERATED`             | LLM-authored                                    |
| 2     | `MACHINE_REFINED`               | Human code an LLM rewrote — mixed authorship    |
| 3     | `MACHINE_GENERATED_ADVERSARIAL` | LLM output prompted to read as human            |

> **The index→label mapping is asserted by the model card, not by the artifact.**
> `config.json` carries no `id2label`, so nothing in the checkpoint pins class order.
> Read the order wrong and `ai_confidence_pct` inverts silently — a fully confident
> human file would score 100. Confirm the order once with a smoke test over a
> handful of unmistakable samples (a few hand-written files, a few freshly
> LLM-generated ones) and assert it in the test suite. This is a correctness check
> on our own output handling, not a re-evaluation of the model.

### Deriving `ai_confidence_pct`

```
ai_confidence_pct = 100 × (1 − P(HUMAN_GENERATED))
```

Rationale: Deltx needs a scalar for *degree of AI involvement*, and all three
machine classes represent AI involvement. Collapsing them to their complement of
class 0 turns the noisy 4-way decision (92.95 F1) into the reliable
human-vs-machine decision (99.18 F1) without discarding the adversarial and
refined classes — a `MACHINE_GENERATED` sample misread as `MACHINE_REFINED` still
scores high, because the confusion is *inside* the collapsed group.

Retain the full 4-way probability distribution on every result object. Stages 3–5
may want the finer signal (e.g. `MACHINE_REFINED` is a plausible quality-decay
predictor in its own right), and discarding it here would be irreversible.

### Internal Pipeline (5 stages)

1. Receive raw Python source files for a commit
2. Tokenize with the ModernBERT tokenizer; files over 8,192 tokens are split into
   overlapping chunks, and their chunk probabilities are combined by token-count
   weighting (the same weighting logic as the file→commit step)
3. Single forward pass per file (or per chunk) → 4-class logits → softmax
4. Collapse to `P(AI) = 1 − P(HUMAN_GENERATED)`
5. Aggregate file → commit by LOC-weighted average → `ai_confidence_pct ∈ [0, 100]`

The score is the model's raw softmax mass, not a calibrated probability — a
cross-entropy + triplet objective gives no calibration guarantee. It is a monotone
confidence signal, which is what Stage 4 needs from an input channel; treat it as
ordinal rather than as a literal percentage likelihood.

### Integration Contract

- **Input:** Raw Python source of every file modified in a commit, plus metadata
  (commit hash, timestamp, author)
- **Output:** `ai_confidence_pct ∈ [0, 100]` where 0 = high confidence human,
  100 = high confidence AI; plus the retained 4-way distribution
- **Granularity:** File-level classification → commit-level LOC-weighted average
- **Skipped files:** non-`.py`, `setup.py`, `conftest.py`, anything under
  `__pycache__`. Unparseable or empty files are excluded from the commit average
  ("assume human when in doubt"); a commit with nothing classifiable scores 0.0
- **Processing:** Offline batch. Target throughput ≥ 50–100 commits/minute
  including overhead. One encoder forward pass per file should clear this
  comfortably — but measure before quoting a figure
- **Downstream consumers:** PatchTST input channel (Stage 4), SHAP feature
  attribution (Stage 5)

> **Detection has no module-local SHAP.** Explainability lives entirely in Stage 5,
> where SHAP attributes over the 15-D commit vector and `ai_confidence_pct` is one
> input feature. TreeExplainer does not apply to a transformer, and Stage 2's job is
> to emit one well-defined scalar, not to explain itself.

## Coding Conventions

- **Type annotations** on all function signatures and return types
- **Pydantic v2** models for all structured data (use `model_validator` for complex validation)
- **Google-style docstrings** on all public functions and classes
- **Minimum 80% test coverage** per module
- **`logging`** module with `rich` handler for structured output; never use `print()`
- **Explicit error handling** — no bare `except:` clauses; define custom exceptions in `common/exceptions.py`
- **`pathlib.Path`** for all file system operations; never use string concatenation for paths
- **Constants** in UPPER_SNAKE_CASE; define in `common/constants.py`
- **No hardcoded model paths or thresholds** — all configurable via `common/config.py`
- **Imports:** standard library → third-party → local, enforced by ruff's isort rules
- **Tests never download the checkpoint.** Mock the detector; mark anything that
  needs real weights `@pytest.mark.slow`.

## Key Terminology

- **Continuous Ordinal Sampling:** Evaluating every sequential commit on the primary branch (never skip commits)
- **15-D Vector:** The 15-dimensional feature vector per commit that feeds PatchTST
- **Squale:** The quality model framework adapted for ISO/IEC 25010 scoring
- **ai_confidence_pct:** The scalar output of the detection module (index [4] of the 15-D vector)
- **DroidDetect:** The pre-trained encoder-only detector Deltx wraps for Stage 2

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
│   ├── parser.py      # Python AST parsing + lexical tokenization
│   ├── features/
│   │   ├── perplexity.py    # F1–F6: surprisal-based features
│   │   ├── stylometric.py   # F7–F12: code style features
│   │   └── distribution.py  # F13–F16: statistical distribution features
│   ├── pipeline.py    # Feature extraction orchestrator
│   ├── classifier.py  # XGBoost/RF training and prediction
│   ├── inference.py   # File → commit-level inference pipeline
│   └── dataset.py     # Dataset download, filtering, preprocessing
├── extraction/        # Stage 1: Data Collection (future)
├── scoring/           # Stage 3: Squale Quality Aggregation (future)
├── prediction/        # Stage 4: PatchTST Forecasting (future)
└── interpretation/    # Stage 5: SHAP Explainability (future)
```

## Technology Stack

- **Python 3.12** with **Poetry** for dependency management
- **PyTorch** + **HuggingFace Transformers** for language model inference
- **XGBoost** as primary classifier (Random Forest as fallback)
- **SHAP** (TreeExplainer) for feature contribution analysis
- **Pydantic v2** for all data models and validation
- **pytest** for testing, **ruff** for linting/formatting, **mypy** for type checking
- **GitHub Actions** for CI/CD

## AI Detection Module — Complete Specification

### Purpose

Addresses the "Invisibility Gap": assigns each commit a probabilistic AI-authorship score (ai_confidence_pct) quantifying the likelihood that code was LLM-generated. This score occupies index [4] of the 15-dimensional data vector as an Evolutionary Driver.

### Internal Pipeline (6 stages)

1. Receive raw Python source files for a commit
2. Tokenize using Python `ast` module + lexical tokenizer → AST + flat token sequence
3. Score tokens against pre-trained code LM → per-token log-probabilities → surprisal values
4. Extract 16 features across three families → fixed-length feature vector
5. Classify via XGBoost/RF trained on labelled human-vs-AI samples → probability
6. Output calibrated ai_confidence_pct, aggregated file→commit via LOC-weighted averaging

### Pre-trained Language Model

**Model:** `Salesforce/codegen-350M-mono` (autoregressive, Python-specific, 350M params)
**Purpose:** Compute token-level log-probabilities for surprisal features F1–F6.
**Rationale:** Autoregressive architecture produces true left-to-right conditional probabilities matching the surprisal formula. Python-specific training improves sensitivity to code-specific patterns. 350M parameters balances quality against batch processing throughput.

### Feature Taxonomy (16 features)

#### Family A — Perplexity & Surprisal (F1–F6)

Surprisal definition: `S(tᵢ) = −log₂ P(tᵢ | t₁, t₂, …, tᵢ₋₁)`

| ID  | Name                    | Definition                                              |
|-----|-------------------------|---------------------------------------------------------|
| F1  | Mean Token Surprisal    | Arithmetic mean of S(tᵢ) across all tokens              |
| F2  | Surprisal Variance      | Variance of per-token surprisal values                  |
| F3  | Sequence Perplexity     | exp(mean(S(tᵢ))); model uncertainty measure             |
| F4  | Max Surprisal           | max(S(tᵢ)); peak anomaly token                         |
| F5  | Low-Surprisal Ratio     | Fraction of tokens where S(tᵢ) < threshold             |
| F6  | Surprisal Slope         | Linear regression slope of S(tᵢ) over token position   |

#### Family B — Stylometric (F7–F12)

| ID  | Name                    | Definition                                              |
|-----|-------------------------|---------------------------------------------------------|
| F7  | Avg Identifier Length   | Mean character length of variable/function/class names  |
| F8  | Identifier Diversity    | Unique identifiers / total identifier count             |
| F9  | Whitespace Consistency  | Std deviation of indentation levels across lines        |
| F10 | Comment-to-Code Ratio   | Comment lines / total lines                             |
| F11 | AST Depth (Mean)        | Average nesting depth of AST nodes                      |
| F12 | AST Node-Type Diversity | Shannon entropy of AST node type frequency distribution |

#### Family C — Distribution (F13–F16)

| ID  | Name                    | Definition                                              |
|-----|-------------------------|---------------------------------------------------------|
| F13 | Shannon Entropy         | H = −∑ p(t) log₂ p(t) over token distribution          |
| F14 | Zipf Coefficient Dev.   | Deviation from expected Zipf exponent in frequency-rank |
| F15 | Bigram Repetition Rate  | Fraction of token bigrams that appear more than once    |
| F16 | Hapax Legomena Ratio    | Fraction of tokens appearing exactly once               |

### Integration Contract

- **Input:** Raw Python source code of every file modified in a commit, plus metadata (commit hash, timestamp, author)
- **Output:** `ai_confidence_pct ∈ [0, 100]` where 0 = high confidence human, 100 = high confidence AI
- **Granularity:** File-level classification → commit-level LOC-weighted average
- **Processing:** Offline batch. Target throughput: 50–100 commits/minute including overhead
- **Downstream consumers:** PatchTST input channel (Stage 4), SHAP feature attribution (Stage 5)

### Training Data Sources

| Dataset         | Size        | Languages    | AI Models                       | Role                        |
|-----------------|-------------|--------------|--------------------------------|-----------------------------|
| DroidCollection | >1M samples | 7 langs      | 43 coding models               | Primary (Python-filtered)   |
| AIGCodeSet      | 7,583       | Python only  | CodeLlama, Codestral, Gemini   | Supplementary               |
| IBM CodeNet     | ~14M        | 55 langs     | Human only                     | Human ground truth          |
| GPTSniffer      | ~5K pairs   | Java, Python | ChatGPT (GPT-3.5)             | Supplementary               |

Target training set: 10,000–20,000 paired samples (5,000–10,000 per class).
Validation: Stratified 5-fold CV with leave-one-model-out held-out test.

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

## Key Terminology

- **Continuous Ordinal Sampling:** Evaluating every sequential commit on the primary branch (never skip commits)
- **15-D Vector:** The 15-dimensional feature vector per commit that feeds PatchTST
- **Squale:** The quality model framework adapted for ISO/IEC 25010 scoring
- **ai_confidence_pct:** The scalar output of the detection module (index [4] of the 15-D vector)

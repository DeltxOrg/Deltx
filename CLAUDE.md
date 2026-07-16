# Deltx — Predictive Software Quality Analytics Platform

## Project Identity

Deltx is a PhD research prototype (with product ambitions) that predicts software quality decay in Python repositories. It combines AI authorship detection, Squale-adapted quality scoring, PatchTST time-series forecasting, and SHAP explainability into a unified pipeline. Every commit in a repository's history is analyzed and encoded as a 15-dimensional vector that feeds the forecasting model.

**Scope constraint:** Python repositories only. No multi-language generalization.

## Monorepo Architecture

Five decoupled modules in a single repository:

```
src/deltx/
├── common/            # Shared data models, config, utilities
├── detection/         # Stage 2: AI Authorship Detection ✓
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
├── scoring/           # Stage 3: Squale Quality Aggregation ✓
│   ├── models.py      # Pydantic models: SonarIssue → CommitQualityVector
│   ├── sonar_client.py # SonarQube Web API client + fixture mode
│   ├── iso_mapping.py  # Rule → ISO/IEC 25010 dimension mapping
│   ├── call_graph.py   # AST call graph + PageRank centrality + churn
│   ├── weighting.py    # Dynamic issue weighting formula
│   ├── scoring.py      # Penalty accumulation + Z-score normalizer
│   ├── aggregation.py  # Squale exponential aggregation
│   ├── pipeline.py     # score_commit() orchestrator + CLI
│   └── tune.py         # Hyperparameter grid search (Spearman)
├── prediction/        # Stage 4: PatchTST Forecasting (future)
└── interpretation/    # Stage 5: SHAP Explainability (future)
```

## Technology Stack

- **Python 3.12** with **Poetry** for dependency management
- **PyTorch** + **HuggingFace Transformers** for language model inference
- **XGBoost** as primary classifier (Random Forest as fallback)
- **SHAP** (TreeExplainer) for feature contribution analysis
- **Pydantic v2** for all data models and validation
- **pandas** + **pyarrow** for dataset construction; **huggingface-hub** for downloads
- **pytest** for testing, **ruff** for linting/formatting, **mypy** for type checking
- **GitHub Actions** for CI/CD

> **pyarrow must be ≥ 24.0.** Earlier releases (16.x confirmed) ship DLLs that break
> `import torch` on Windows with `WinError 1114` on `c10.dll`. Because pandas
> auto-imports pyarrow when installed, `import pandas` alone is enough to trigger it.
> Likewise `huggingface-hub` is floored, not capped: capping it pins `transformers`
> to an old release.

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
| F3  | Sequence Perplexity     | 2 ** mean(S(tᵢ)); model uncertainty measure (bits base) |
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

Implemented in `detection/dataset.py`. Every origin below was verified against the
live publisher; the counts are what actually ships.

| Source key        | Origin                                    | Python samples          | Role                      |
|-------------------|-------------------------------------------|-------------------------|---------------------------|
| `droidcollection` | HF `project-droid/DroidCollection`        | ~262k (train split)     | Primary (Python-filtered) |
| `aigcodeset`      | HF `basakdemirok/AIGCodeSet`              | 4,755 human + 2,828 AI  | Supplementary             |
| `codenet`         | IBM Project CodeNet, `Python800` subset   | 240,000 (human only)    | Human ground truth        |
| `gptsniffer`      | GitHub `MDEGroup/GPTSniffer`              | **none** — Java only    | Unusable, see below       |

**DroidCollection** (`project-droid/DroidCollection`, EMNLP 2025). Parquet shards
under `data/`, ~1.06M rows across train/dev/test and **9 languages**.
Columns: `Code`, `Label`, `Language`, `Generator`, `Model_Family`,
`Generation_Mode`, `Source`, and two parameter blobs. The `Generator` column holds
45 distinct model names plus the literal `Human`.

Its `Label` column is **four-class, not binary**:

| Label                           | Maps to    | Rationale                                              |
|---------------------------------|------------|--------------------------------------------------------|
| `HUMAN_GENERATED`               | `label=0`  |                                                        |
| `MACHINE_GENERATED`             | `label=1`  |                                                        |
| `MACHINE_REFINED`               | *dropped*  | Human code an LLM rewrote — mixed authorship           |
| `MACHINE_GENERATED_ADVERSARIAL` | *dropped*  | AI output styled to read as human; would poison class 0 |

`DatasetManager.DROID_LABEL_MAP` encodes this policy; override the class attribute
to change it. The two dropped classes are ~25% of Python rows (measured on the dev
split: 24,641 of 32,761 kept).

**AIGCodeSet** (`basakdemirok/AIGCodeSet`). The GitHub repository referenced in
earlier drafts of this document does not exist; the HuggingFace mirror is
authoritative. Two CSVs are read (`data/human_selected_dataset.csv`,
`data/created_dataset_with_llms.csv`); a third combined CSV carrying ada embeddings
is 265 MB and deliberately skipped. AI half generated by CodeLlama 34B, Codestral
22B and Gemini 1.5 Flash (`ai_model` ∈ `llama`, `codestral`, `gemini`).

Its human half is drawn from CodeNet, and the overlap is measurable: **451 of its
4,755 human rows (9.5%) are byte-identical to `Python800` submissions**. Dedup is
therefore load-bearing whenever both sources are enabled.

> **Label-conflicting duplicates.** AIGCodeSet contains **103 code strings that
> carry both `label=0` and `label=1`** — the LLM reproduced a human solution
> verbatim, so the same bytes appear in both CSVs. These are contradictions, not
> duplicates. `_drop_label_conflicts()` removes *every* copy of such a string
> (206 rows) before deduplication runs.
>
> Resolving them by keeping one copy would let file-read order assign ground
> truth: the AI copy would win purely because `created_dataset_with_llms.csv`
> sorts before `human_selected_dataset.csv`. Both copies go instead — a string
> that demonstrably belongs to both classes teaches the classifier nothing.
>
> After conflict removal, dedup drops a further 27 genuinely benign duplicates
> (5 within the human half, 22 within the AI half). Loading AIGCodeSet alone
> therefore yields 7,337 rows: 4,636 human and 2,701 AI.

**IBM CodeNet.** The full archive is 7.8 GB across 55 languages and ~14M
submissions. Deltx downloads only the `Project_CodeNet_Python800` benchmark
subset: a 30 MB tarball of 240,000 Python files across 800 problems, all
human-written (`label=0`).

**GPTSniffer** (Nguyen et al., JSS 2024). Its replication package contains 28,174
Java files and 26 Python files, and all 26 are the tool's own source code rather
than samples. It therefore contributes **zero** Python training samples. Since
Deltx is Python-only, `download_gptsniffer()` fetches nothing and instead writes
placement instructions for a `human/` + `ai/` directory layout that
`load_from_directory("gptsniffer")` will read if data is supplied by hand.

**Unified schema** produced by `load_and_unify()`: `source_code`, `label` (0=human,
1=AI), `source_dataset`, `ai_model` (`None` for human rows), `language`. Filters run
in this order, and the order matters:

1. Python only
2. Minimum 10 tokens
3. Drop *all* copies of any `source_code` carrying more than one `label`
4. Drop exact-duplicate `source_code` (first source listed wins a collision)

Step 3 must precede step 4, or deduplication collapses the disagreeing rows and
hides the conflict. Because it runs first, every collision surviving into step 4
agrees on its label, so load order cannot change any sample's ground truth.

Target training set: 10,000–20,000 paired samples (5,000–10,000 per class); use
`load_and_unify(max_per_source=...)` to cap the large corpora, since scoring every
sample against the 350M-parameter LM dominates pipeline cost.
Validation: Stratified 5-fold CV with leave-one-model-out held-out test
(`prepare_train_test_split(holdout_model=...)`; the match is exact and
case-insensitive, so `"llama"` never sweeps in `"codellama"`).

## Quality Scoring Module — Complete Specification

### Purpose

Translates raw SonarQube rule violations at a given commit into four standardized ISO/IEC 25010 scores — `score_maintainability`, `score_correctness`, `score_security`, `score_efficiency` — on a 0–100 scale (100 = perfect). Uses dynamic issue weighting and Squale-inspired exponential penalty aggregation. Runs in parallel with Stage 2 (AI Detection) — no dependency on `ai_confidence_pct`.

### Mathematical Specification

#### Per-issue dynamic weight

For issue `i` with severity `S_i`, local frequency `f_i`, centrality `C_i ∈ [0,1]`, and churn `K_i`:

```
w_i = S_i · (1 + α · ln(1 + f_i)) · (1 + β · C_i) · (1 + γ · K_i)
```

Default hyperparameters: `α=0.5, β=1.0, γ=0.3`. Tunable via `tune.py` grid search.

#### Dimension penalty density

```
P_d = (Σ w_i for i ∈ dimension d) / LOC_active
```

#### Z-score normalization and inversion

```
z_d   = (P_d − μ_d) / σ_d        # μ_d, σ_d persisted from training
score = 100 · (1 − clip(minmax(z_d), 0, 1))
```

#### Squale exponential aggregation (system-level)

```
Score_d = −100 · ln(mean(λ^(−s_m / 100))) / ln(λ)
```

Default `λ=30.0`. Dominated by the worst module: a single critical defect in a core routing module collapses the global score.

### ISO/IEC 25010 Dimension Mapping

| SonarQube Type      | Default Dimension | Override Rules                         |
|---------------------|-------------------|----------------------------------------|
| BUG                 | correctness       | —                                      |
| VULNERABILITY       | security          | —                                      |
| SECURITY_HOTSPOT    | security          | —                                      |
| CODE_SMELL          | maintainability   | Efficiency/correctness rule overrides  |

Severity scores: BLOCKER=10, CRITICAL=7, MAJOR=4, MINOR=2, INFO=1.

### Integration Contract

- **Input:** SonarQube issues + measures for a commit, source tree for call graph, Git repo for churn
- **Output:** `CommitQualityVector` with four floats keyed as `score_maintainability`, `score_correctness`, `score_security`, `score_efficiency` — all in [0, 100]
- **Granularity:** Per-file module scoring → Squale system-level aggregation
- **CLI:** `deltx-score --from-fixture issues.json --src ./checkout --commit SHA`
- **Downstream consumers:** PatchTST target channels (Stage 4), SHAP attribution (Stage 5)

### 15-D Vector Field Mapping

The canonical 15-D vector (`CommitDataVector` in `deltx.common.models`) carries:

| Index | Field Name                | Source Module |
|-------|---------------------------|---------------|
| 0     | `commit_size`             | extraction    |
| 1     | `file_count`              | extraction    |
| 2     | `complexity_delta`        | extraction    |
| 3     | `churn_rate`              | extraction    |
| 4     | `ai_confidence_pct`       | detection     |
| 5     | `score_maintainability`   | **scoring**   |
| 6     | `score_correctness`       | **scoring**   |
| 7     | `score_security`          | **scoring**   |
| 8     | `score_efficiency`        | **scoring**   |
| 9     | `author_experience`       | extraction    |
| 10    | `time_since_last_commit`  | extraction    |
| 11    | `test_coverage_delta`     | extraction    |
| 12    | `dependency_count_delta`  | extraction    |
| 13    | `documentation_ratio`     | extraction    |
| 14    | `coupling_score`          | extraction    |

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

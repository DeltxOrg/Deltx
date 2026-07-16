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
│   ├── modeling.py    # TLModel architecture + checkpoint loading
│   ├── detector.py    # DroidDetect wrapper: source → class probabilities
│   ├── inference.py   # File → commit-level inference pipeline
│   └── cli.py         # Command-line interface (deltx-detect)
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

Detection is performed by **DroidDetect-Base-Binary**, a published pre-trained
AI-generated-code detector, used directly as the Stage 2 detector. Deltx neither
trains nor re-evaluates a detector: the model is already trained, benchmarked, and
peer-reviewed by its authors, and Deltx consumes it as a fixed component. All
engineering effort goes into the wrapper — loading the checkpoint correctly, reading
its outputs correctly, and aggregating file-level predictions to the commit level.

### Detection Model

**Model:** `project-droid/DroidDetect-Base-Binary` (HuggingFace), Apache-2.0.
**Paper:** Orel, Paul et al., *Droid: A Resource Suite for AI-Generated Code
Detection*, EMNLP 2025 Main ([arXiv:2507.10583](https://arxiv.org/abs/2507.10583)).

| Property           | Value                                                          |
|--------------------|----------------------------------------------------------------|
| Backbone           | `answerdotai/ModernBERT-base` (encoder-only, 149M params)       |
| Head               | mean-pool → `Linear(768 → 256)` → ReLU → `Linear(256 → 2)`      |
| Task               | Binary classification: human vs. machine                        |
| Context window     | 8,192 tokens (ModernBERT native) — most source files fit whole   |
| Training corpus    | Filtered `project-droid/DroidCollection` (7 languages incl. Python) |
| Training objective | `CrossEntropyLoss + 0.1 × BatchHardSoftMarginTripletLoss`        |

The published suite offers binary, ternary, and four-class heads in Base and Large
sizes. Deltx takes the **binary Base** checkpoint on two grounds. The binary setup
is by a wide margin the most reliable — **99.18 weighted F1**, against 94.36 for
ternary and 92.95 for four-class — and Stage 2 needs exactly one scalar, so the
finer heads' extra classes would be collapsed on arrival anyway. Taking the binary
head moves that collapse into the model's own training, where machine-refined code
is folded in with generated code as a training target rather than summed after the
fact. Large trails Base by well under a point (99.25 vs 99.18) at ~2.7× the size,
and Stage 2 runs over every commit in a repository's history.

**In-domain weighted F1, as reported in the paper.** These are the authors'
published numbers and are the basis on which Deltx adopts the model; Deltx does not
reproduce them.

| Variant                  | 2-class |
|--------------------------|---------|
| DroidDetect-Base-Binary  | 99.18   |
| DroidDetect-Large-Binary | 99.25   |

### Checkpoint Loading Contract

The HF repo ships exactly: `config.json`, `pytorch_model.bin`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`.
There is **no modeling module and no `auto_map`**, and `config.json` is minimal:

```json
{"model_type": "custom_model", "architectures": ["Model"],
 "projection_dim": 128, "num_classes": 2}
```

> **`projection_dim` in `config.json` is wrong.** It says `128`, and so does the
> model card's `TLModel.__init__` default, but the shipped weights are
> `text_projection.weight (256, 768)` and `classifier.weight (2, 256)`. The real
> head is **768 → 256 → 2**. Build to the weights, not to the config; instantiating
> at 128 fails with a shape mismatch. Verified by inspecting `pytorch_model.bin`
> directly — do not trust either published value. The same error appears verbatim
> across the suite's checkpoints, so it is a property of how they were exported,
> not a one-off typo.

The checkpoint's actual contents, measured rather than assumed: **272 tensors under
exactly four prefixes** — `text_encoder.*` (134), `additional_loss.*` (134),
`text_projection.*` (2), `classifier.*` (2).

Consequences, all load-bearing:

1. **`AutoModelForSequenceClassification.from_pretrained` cannot load this
   checkpoint.** `modeling.py` must define the `TLModel` architecture locally,
   instantiate `ModernBERT-base` as the `text_encoder`, then load
   `pytorch_model.bin` into it with `torch.load`.
2. **The encoder keys match upstream `answerdotai/ModernBERT-base` exactly** —
   134 tensors, zero missing, zero unexpected. So `AutoModel.from_config` on the
   upstream config gives a container the checkpoint drops straight into.
3. **`additional_loss.*` ships and must be filtered out.** The training-time
   `BatchHardSoftMarginTripletLoss` held a reference to the encoder, so the export
   captured a second full ModernBERT under
   `additional_loss.sentence_embedder.*` — 134 tensors that inference has no use
   for. These are *aliases*, not a second copy: the file is 569 MB, byte-identical
   in size to a checkpoint carrying one backbone, because `torch.save` serialises
   shared storages once. Dropping the prefix discards nothing.
   Filter that one known prefix and load the remaining **138 with `strict=True`**;
   do not reach for a blanket `strict=False`. Strictness over what remains is what
   guarantees the projection and classifier weights actually came from the
   checkpoint rather than staying at their random initialisation, and it will still
   catch any future revision that changes shape.
4. **Do not add `sentence-transformers` as a runtime dependency.** It is needed
   only to construct the training-time loss, which inference omits entirely —
   the loss's serialised weights are dropped at load time.
5. **The tokenizer loads normally** via `AutoTokenizer.from_pretrained`:
   `PreTrainedTokenizerFast`, vocab 50,280, `model_max_length` 8192, `[PAD]` /
   `[CLS]` / `[SEP]` all present. It wraps input in `[CLS]`/`[SEP]`, and those
   positions enter the mean pool — which is what training did, so keep it.

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

| Index | Label               | Meaning                                              |
|-------|---------------------|------------------------------------------------------|
| 0     | `HUMAN_GENERATED`   | Human-authored                                        |
| 1     | `MACHINE_GENERATED` | Any AI involvement — generated, refined, or adversarial |

Class 1 is deliberately broad. The paper's binary setup maps the ternary labels
(human-written, AI-generated, AI-refined) onto two targets, so a human file an LLM
rewrote is a class-1 training example. `MACHINE_REFINED` is therefore not a class
Deltx discards — it is one the model was trained to report as machine.

**Verify this order against the artifact; do not take it from the card.**
`config.json` carries no `id2label`, and the model card that states
`{"0": "HUMAN_GENERATED", "1": "MACHINE_GENERATED"}` is the same card that
misreports `projection_dim`, so it is corroboration rather than proof. The test
suite pins the order behind `@pytest.mark.slow`: complete pre-LLM stdlib modules
must land on index 0 and known-LLM source on index 1, separated by a wide margin
— a near-boundary pass on both would indicate a reordered head or an unloaded
classifier. Read the order wrong and `ai_confidence_pct` inverts silently while
every other test still passes.

> **Feed complete files — never short fragments.** This is the sharpest practical
> constraint on the module. Fifteen complete stdlib modules score a median P(AI)
> of **0.198**, 13/15 correctly human. Forty individual methods pulled from eight
> of those same modules via `inspect.getsource` score a median of **0.604**, with
> **23/40** over the 0.5 threshold and a maximum of 0.996. Identical code, a 3×
> higher median and a 4.5× higher false-positive rate, purely from being served in
> pieces.
>
> **The cause is decontextualization, not indentation.** Scoring each fragment a
> second time with `textwrap.dedent` applied — same code, no leading whitespace —
> moves the median *up* to **0.831** and the false-positive count to **29/40**.
> Stripping the indentation makes things worse, so do not reach for dedenting as a
> mitigation. What the model reacts to is a short, self-contained-looking unit torn
> out of its surroundings, which is exactly what an LLM is usually asked to emit.
>
> Two consequences. First, this is independent support for whole-file scoring over
> diff hunks: hunks are exactly the decontextualized fragments that fail. Second,
> chunking past the context window is **not** free — every chunk after the first is
> a fragment, so `split_into_chunks` cuts on top-level boundaries to keep chunks as
> large and self-contained as possible, and treats a chunk that could not be
> top-level-aligned as suspect. Prefer fewer, larger chunks over more, smaller ones.
>
> Residual false positives on whole human files remain: of 15 stdlib modules,
> `heapq.py` scored 0.769 and `queue.py` 0.540. Expect roughly this rate of noise
> on real repositories.

### Deriving `ai_confidence_pct`

```
ai_confidence_pct = 100 × (1 − P(HUMAN_GENERATED))
```

Over two classes this equals `100 × P(MACHINE_GENERATED)`. It is written as the
complement so the definition stays anchored to class 0 — the class whose meaning
is fixed and directly verified — rather than to whatever the machine class happens
to encompass.

Rationale: Deltx needs one scalar for *degree of AI involvement*, and the binary
head emits exactly that at the detector's most reliable operating point (99.18
weighted F1). Because the model was trained with refined code folded into class 1,
no confusion between kinds of AI involvement can move this number; the collapse
happens during training rather than in post-processing.

Retain the full probability distribution on every result object rather than only
the scalar. It costs nothing, keeps the result self-describing for Stages 3–5, and
means a future switch to a finer-grained head changes the distribution's width
without changing the shape of the contract.

> **Stages 3–5 get no `MACHINE_REFINED` channel.** The binary head does not
> distinguish refined from generated code, so a "was this human code an LLM
> rewrote?" feature is not available downstream and cannot be recovered from a
> stored result. This is a deliberate trade: the finer signal is speculative,
> while the binary head's ~6-point F1 advantage over the four-class head is
> measured. If a later stage establishes that refinement predicts decay
> independently, that is a reason to revisit the checkpoint choice — not to
> post-process this score.

### Internal Pipeline (5 stages)

1. Receive raw Python source files for a commit
2. Tokenize with the ModernBERT tokenizer; files over 8,192 tokens are split into
   overlapping chunks, and their chunk probabilities are combined by token-count
   weighting (the same weighting logic as the file→commit step)
3. Single forward pass per file (or per chunk) → 2-class logits → softmax
4. Read `P(AI) = 1 − P(HUMAN_GENERATED)`
5. Aggregate file → commit by LOC-weighted average → `ai_confidence_pct ∈ [0, 100]`

The score is the model's raw softmax mass, not a calibrated probability — a
cross-entropy + triplet objective gives no calibration guarantee. It is a monotone
confidence signal, which is what Stage 4 needs from an input channel; treat it as
ordinal rather than as a literal percentage likelihood.

### Integration Contract

- **Input:** Raw Python source of every file modified in a commit, plus metadata
  (commit hash, timestamp, author)
- **Output:** `ai_confidence_pct ∈ [0, 100]` where 0 = high confidence human,
  100 = high confidence AI; plus the retained 2-class distribution
- **Granularity:** File-level classification → commit-level LOC-weighted average
- **Skipped files:** non-`.py`, `setup.py`, `conftest.py`, anything under
  `__pycache__`. Unparseable or empty files are excluded from the commit average
  ("assume human when in doubt"); a commit with nothing classifiable scores 0.0
- **Processing:** Offline batch. **Measured: 7.6 files/min ≈ 2.5 commits/min on
  CPU** (Intel CPU, 8 real stdlib files, 343 tokens/s). That is 20–40× short of
  the 50–100 commits/min target, which is therefore *not* met on CPU and should
  not be quoted as if it were
- **Downstream consumers:** PatchTST input channel (Stage 4), SHAP feature
  attribution (Stage 5)

> **Detection has no module-local SHAP.** Explainability lives entirely in Stage 5,
> where SHAP attributes over the 15-D commit vector and `ai_confidence_pct` is one
> input feature. TreeExplainer does not apply to a transformer, and Stage 2's job is
> to emit one well-defined scalar, not to explain itself.

> **Bulk historical runs need a GPU.** Cost grows superlinearly with file length —
> a 1,507-token file took 3.1 s while a 6,086-token file took 23.7 s, so quadratic
> attention dominates and large files are disproportionately expensive. At the
> measured CPU rate, a 1,000-commit history is roughly **6–7 hours**. The
> development machine has no CUDA device (Intel Iris Xe), so the module is kept
> importable and free of CLI-only state specifically so bulk scoring can run in a
> GPU notebook, with the local CPU path reserved for development and small
> repositories.

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
- **Tests never download the checkpoint.** Mock the detector; mark anything that
  needs real weights `@pytest.mark.slow`. The marker only labels — `addopts`
  carries `-m 'not slow'` to actually deselect them, and removing it silently
  puts a 569 MB download back in the default suite.

## Key Terminology

- **Continuous Ordinal Sampling:** Evaluating every sequential commit on the primary branch (never skip commits)
- **15-D Vector:** The 15-dimensional feature vector per commit that feeds PatchTST
- **Squale:** The quality model framework adapted for ISO/IEC 25010 scoring
- **ai_confidence_pct:** The scalar output of the detection module (index [4] of the 15-D vector)
- **DroidDetect:** The pre-trained encoder-only detector Deltx wraps for Stage 2

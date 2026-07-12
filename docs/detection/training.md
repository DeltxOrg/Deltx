# Training the AI-Authorship Detector — Production Pipeline

This guide is for developers who need to (re)train the Stage-2 detector on the
full labelled corpora and ship a `detector.joblib` the rest of Deltx can load. It
covers the three-phase pipeline, how to run each phase, and every tunable knob
that actually exists.

For the *module* internals (features, scoring, inference) see
[README.md](README.md). For a one-command sanity check use
`scripts/validate_pipeline.py` — that is a 100-sample smoke test, **not** a
production trainer; this document describes the real thing.

## Why three phases

Training decomposes into three steps with very different costs and hardware
needs, so they are three separate artifacts joined by two parquet files:

```
Phase A  scripts/build_training_set.py     LOCAL  (CPU/network, minutes)
            └─ data/processed/train_balanced.parquet   (balanced text corpus)
Phase B  notebooks/extract_features_gpu.ipynb   CLOUD GPU (~20–40 min on a T4)
            └─ data/processed/train_features.parquet   (16-D feature matrix)
Phase C  scripts/train_detector.py         LOCAL  (CPU, minutes)
            └─ data/models/detector.joblib             (shipped classifier)
```

The expensive step is **Phase B**: every sample is scored against the
350M-parameter CodeGen language model to produce the surprisal features. That is
the *only* step that benefits from a GPU. Balancing happens in Phase A —
**before** extraction — so the GPU never spends time on samples that would later
be discarded to balance the classes. Once features exist they are model-agnostic
numbers, so Phase C (XGBoost training + SHAP) runs comfortably on CPU.

> **Fully local alternative.** If you have no GPU, skip the notebook and run the
> same extraction call locally with `DELTX_DEVICE=cpu`. Budget roughly
> **2–3 seconds per sample** on a modern laptop CPU (≈10–15 h for a 20k corpus).
> Extraction checkpoints every 500 rows and resumes, so an overnight run is safe.

## Prerequisites

- Python 3.12 and `poetry install` (see the repo README).
- `pyarrow >= 24` — already pinned; older DLLs break `import torch` on Windows.
- Disk: ~700 MB for the CodeGen model, plus a few GB for the raw corpora
  (CodeNet's `Python800` tarball alone is ~30 MB extracted to 240k files).
- For Phase B, a free cloud GPU account:
  - **Kaggle Notebooks** (recommended) — free T4, 30 GPU-h/week, stable sessions.
  - **Google Colab (free)** — free T4, but sessions can idle-disconnect.
  - No paid tier is required; the workload fits the free limits comfortably.

## Phase A — assemble a balanced corpus

`scripts/build_training_set.py` downloads the three usable sources (AIGCodeSet,
DroidCollection, CodeNet — GPTSniffer is excluded, it ships zero Python samples),
unifies and deduplicates them via `DatasetManager.load_and_unify`, then
downsamples to a **balanced 50/50 human/AI** corpus.

```bash
# Full build (downloads on first run):
poetry run python scripts/build_training_set.py

# Fast smoke test — AIGCodeSet only, ~200 rows, no large downloads:
poetry run python scripts/build_training_set.py --sources aigcodeset --limit 200
```

Output: `data/processed/train_balanced.parquet` (unified schema: `source_code`,
`label`, `ai_model`, `source_dataset`, `language`). The script prints a table of
AI generators present in the balanced set — **note one with a healthy count
(~500+ rows); you will pass it as the leave-one-model-out holdout in Phase C.**

The target is deliberately over-provisioned (`--per-class` default 11,000) so the
final ~10k/class survives the rows Phase B rejects (unparseable code / a feature
family failing).

| Flag | Default | Purpose |
|------|---------|---------|
| `--sources` | all three | Subset of `aigcodeset droidcollection codenet` |
| `--per-class` | `11000` | Rows per class after balancing (clamped to the scarcer class) |
| `--max-per-source` | `40000` | Per-source cap applied **before** filtering; bounds memory against DroidCollection's ~262k rows |
| `--output` | `data/processed/train_balanced.parquet` | Where to write |
| `--skip-download` | off | Assume raw sources already present |
| `--limit N` | — | Smoke-test shortcut: caps each source at `N` and sets per-class to `N//2` |

> CodeNet is human-only and its first load reads ~240k small `.py` files (a few
> minutes of disk I/O). If you only need to iterate on Phase C quickly, build with
> `--sources aigcodeset droidcollection` to skip it.

## Phase B — extract features on a GPU

Open `notebooks/extract_features_gpu.ipynb` in Kaggle or Colab with a **GPU
runtime enabled**, then run the cells top to bottom. The notebook:

1. Pulls the Deltx source (edit the clone URL to your repo) and installs
   `pyarrow>=24` + `transformers` — it deliberately does **not** reinstall
   PyTorch, to keep the platform's CUDA build.
2. Sets `DELTX_DEVICE=cuda` **before** `DeltxConfig` is constructed
   (`PerplexityExtractor` reads it, moves the model to CUDA, runs fp16).
3. Reads your uploaded `train_balanced.parquet` and runs
   `DatasetManager.extract_features_dataset(..., checkpoint_every=500)`.
4. Sanity-checks the matrix and helps you download
   `train_features.parquet`.

Bring that parquet back to `data/processed/` locally. Because extraction
checkpoints and resumes, a disconnect just means re-running the extraction cell.

## Phase C — train, evaluate, and ship

`scripts/train_detector.py` reads the feature matrix and does three things:

1. **Headline (in-distribution) evaluation** — a stratified train/val/test split,
   `RandomizedSearchCV` hyperparameter tuning with early stopping, reporting
   hold-out metrics, the 5-fold CV score, a confusion matrix, and SHAP importances.
2. **Leave-one-model-out (LOMO) evaluation** — a *fresh* model trained on every
   generator except `--holdout-model`, then scored on that unseen generator's
   samples only. This measures whether the detector flags an LLM it never saw.
3. **Ships the model** — retrains on the full feature set with the tuned
   parameters and saves to `data/models/detector.joblib` (`config.classifier_path`).

```bash
# Full run with a leave-one-model-out test on the 'gemini' generator:
poetry run python scripts/train_detector.py --holdout-model gemini

# Fast dry run: default hyperparameters, no LOMO:
poetry run python scripts/train_detector.py --no-tune
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--features` | `data/processed/train_features.parquet` | Feature matrix from Phase B |
| `--holdout-model` | — | Generator (`ai_model` value) held out for the LOMO test; omit to skip LOMO |
| `--per-class` | min class | Rebalance to exactly this many rows per class |
| `--no-tune` | off | Skip the hyperparameter search (default params, much faster) |

Verify the shipped model end to end:

```bash
poetry run deltx-detect analyze --file some_module.py   # → ai_confidence_pct in [0, 100]
```

## Tweaks and tuning knobs

Everything below is a real, supported lever. They fall into three tiers by how
much you have to touch.

### 1. Command-line flags (no code, per run)

The `--*` flags in the Phase A and Phase C tables above. The most impactful:
`--per-class` (dataset size / extraction cost), `--holdout-model` (which
generalization test), and `--no-tune` (skip the search for speed).

### 2. Environment variables (no code — `DELTX_`-prefixed `DeltxConfig` fields)

Set inline or in a `.env` file. The ones that matter for training:

| Variable | Default | Effect on training |
|----------|---------|--------------------|
| `DELTX_DEVICE` | `auto` | `cuda` / `cpu` for the extraction LM (Phase B). `auto` picks CUDA when available |
| `DELTX_RANDOM_SEED` | `42` | Seeds sampling, the CV folds, model fit, and SHAP subsampling — set for reproducibility or to vary runs |
| `DELTX_CLASSIFIER_PATH` | `data/models/detector.joblib` | Where Phase C writes the shipped model (e.g. keep experiment variants side by side) |
| `DELTX_MODEL_NAME` | `Salesforce/codegen-350M-mono` | Swap the surprisal-scoring LM (changes F1–F6; requires re-extraction) |
| `DELTX_MODEL_CACHE_DIR` | `data/models/codegen` | LM cache location (e.g. a Kaggle working dir) |
| `DELTX_MAX_SEQUENCE_LENGTH` | `1024` | Token truncation length for LM scoring — longer captures more of big files at higher cost |
| `DELTX_LOW_SURPRISAL_THRESHOLD` | `2.0` | Bits threshold for the F5 feature (changes extraction output) |
| `DELTX_CONFIDENCE_THRESHOLD` | `0.5` | Decision boundary used by `predict`/`evaluate`; raise it to trade recall for precision |

> `DELTX_BATCH_SIZE` exists but is currently **not used** during feature
> extraction — the LM scores one file per forward pass. See "Known limits" below.

### 3. Source constants (require editing `src/`)

These are module-level constants — change them in code, not at runtime.

- **Hyperparameter search** (`detection/classifier.py`): `SEARCH_SPACE` (the grid
  sampled), `SEARCH_N_ITER` (50 candidates), `SEARCH_CV_FOLDS` (5), `SEARCH_SCORING`
  (`"f1"`), and `EARLY_STOPPING_ROUNDS` (20). Widen the space or raise `n_iter` for
  a more thorough search at more CPU cost; change `SEARCH_SCORING` to optimize a
  different metric. `DEFAULT_PARAMS` is what `--no-tune` uses.
- **DroidCollection label policy** (`detection/dataset.py`): `DROID_LABEL_MAP`
  keeps only `HUMAN_GENERATED → 0` and `MACHINE_GENERATED → 1`, dropping
  `MACHINE_REFINED` and `MACHINE_GENERATED_ADVERSARIAL` (~25% of Python rows) as
  mixed/poisoning authorship. Override the class attribute (e.g. on a subclass) to
  admit them if your research needs those classes.
- **Extraction checkpoint interval**: `extract_features_dataset(...,
  checkpoint_every=500)` — lower it for more frequent saves on a flaky GPU session,
  raise it to reduce write overhead.

### Known limits (not currently tunable without new code)

Being honest about the ceilings so nobody hunts for a flag that doesn't exist:

- **No class-imbalance weighting.** There is no `scale_pos_weight`/`class_weight`;
  balance is handled by downsampling in the scripts. Train on a balanced corpus.
- **XGBoost is CPU-only in code.** No `device='cuda'` path for the classifier —
  it does not need one (training is seconds–minutes), but you cannot GPU-accelerate
  it without editing `classifier.py`.
- **No batched LM inference.** `config.batch_size` is dead; extraction scores one
  file per forward pass, so even on a GPU throughput is one-file-at-a-time. This is
  the main efficiency ceiling on Phase B and would require a `pipeline.py` change to
  lift.

## Data notes worth knowing

- **Balance is on you.** `DatasetManager` never rebalances; with CodeNet
  (human-only) + DroidCollection + AIGCodeSet the raw corpus skews human-heavy.
  Phase A downsamples to 50/50; Phase C rebalances again after rejects.
- **Deduplication is order-dependent and load-bearing.** AIGCodeSet's human half
  overlaps CodeNet (451 byte-identical rows), and AIGCodeSet carries 103
  label-conflicting strings that are dropped entirely before dedup. `load_and_unify`
  handles all of this; the source order (AIGCodeSet first) decides collision winners.
- **`ai_model` powers LOMO.** DroidCollection's `Generator` and AIGCodeSet's `LLM`
  columns are recorded (lower-cased) as `ai_model`, so `--holdout-model` can target
  any real generator. Human rows have `ai_model = None` and never match.

## Full end-to-end recipe

```bash
# 1. Assemble balanced corpus (local)
poetry run python scripts/build_training_set.py
#    → note a well-represented generator from the printed table, e.g. "gemini"

# 2. Extract features on a GPU (Kaggle/Colab)
#    run notebooks/extract_features_gpu.ipynb, upload train_balanced.parquet,
#    download train_features.parquet back to data/processed/

# 3. Train, evaluate (in-distribution + LOMO), and ship (local)
poetry run python scripts/train_detector.py --holdout-model gemini

# 4. Confirm the shipped model loads and scores
poetry run deltx-detect analyze --file some_module.py
```

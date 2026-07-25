# Constructing the Deltx AI-Authorship Training Corpus

This document describes the procedure by which the Stage-2 (AI Authorship
Detection) training corpus is assembled: the source corpora, the sampling caps,
the filtering and deduplication pipeline, the observed conflict/duplicate counts,
the resulting class balance, and the losses incurred during feature extraction.

Every figure below is measured from the actual build artifacts, not estimated.

> **Two builds are documented here.** §2–§6 describe the **current** reference
> build — **DroidCollection only, 100,000 rows**. §8 preserves the **superseded**
> three-source build (22,000 rows), because the ten runs recorded in
> `data/runs/index.jsonl` were all produced against it and remain citable only if
> its construction stays on record.

## 1. Design goals

The corpus is built to satisfy four constraints:

1. **Python only.** Deltx targets Python repositories exclusively, so every
   non-Python sample is discarded regardless of source.
2. **Single corpus.** All samples come from one source corpus. Pooling corpora
   lets the classifier separate *corpora* instead of *authorship* — see §1.1.
3. **Class balance.** The downstream classifier is trained without any
   class-imbalance weighting, so the corpus is downsampled to an exact **50/50**
   human/AI ratio before any feature extraction is spent on it.
4. **Provenance integrity.** Label-conflicting samples are removed before
   balancing, so no sample's ground-truth label is decided by chance.

The unified representation used throughout is a five-column schema:
`source_code`, `label` (0 = human, 1 = AI), `source_dataset`, `ai_model`
(the generating model for AI rows, `None` for human rows), and `language`.

### 1.1 Why a single corpus

Constraint 2 replaced the earlier multi-source design, and it was an empirical
decision rather than a stylistic one. Three measurements drove it:

**Feature importance tracked corpus identity, not authorship.**
`f7_avg_identifier_length` was the top-ranked feature by mean |SHAP| in the
pooled build (**0.925**), and collapsed to rank 6 (**0.186**) the moment CodeNet
was removed — while the features that replaced it at the top barely moved.
CodeNet is competitive-programming code, where identifiers are `n`, `i`, `dp`.
The pooled model had learned *"short identifiers ⇒ human"*, which is true of
CodeNet specifically and not of human code generally.

**Generalization to unseen generators improved when the shortcut was removed.**
Leave-one-model-out recall on the held-out `gemini` generator was **0.179**
pooled, and **0.547** with CodeNet dropped. The pooled detector was substantially
blind to an unseen LLM because a corpus shortcut was available instead.

**Cross-corpus transfer is near chance.** Training on DroidCollection and testing
on AIGCodeSet gives AUROC **0.581**; the reverse gives **0.591**
(`transfer-droid-to-aig`, `transfer-aig-to-droid`). Whatever a pooled model
learns does not survive a corpus change, so pooling buys apparent diversity
without buying transferable signal.

DroidCollection is the chosen corpus because it is the only source carrying both
classes in volume from a shared distribution, and it spans 45 generators, so
generator diversity is preserved without pooling.

See [ablation.md](ablation.md) for the feature-family analysis that runs on this
corpus and the reasoning behind its single-corpus design.

## 2. Source corpus and label policy

**DroidCollection** (`project-droid/DroidCollection`, EMNLP 2025) — parquet
shards under `data/`, ~1.06M rows across 9 languages. Filtering to Python yields
**328,383** rows.

Its `Label` column is four-class, and only two map cleanly onto the
`label ∈ {0, 1}` contract:

| DroidCollection label | Rows (Python) | Share | Policy |
|---|---:|---:|---|
| `HUMAN_GENERATED` | 148,211 | 45.1% | → `label = 0` |
| `MACHINE_GENERATED` | 98,196 | 29.9% | → `label = 1` |
| `MACHINE_REFINED` | 46,379 | 14.1% | **dropped** — human code an LLM rewrote; mixed authorship |
| `MACHINE_GENERATED_ADVERSARIAL` | 35,597 | 10.8% | **dropped** — AI styled to read as human; would poison class 0 |
| **Total** | **328,383** | 100% | |

Admitting the adversarial rows in particular would teach the classifier that
human style *is* AI style, which is the one lesson it must not learn.

Applying the policy discards **81,976 rows (25.0%)** and leaves **246,407**
usable Python samples: **148,211 human (60.1%) / 98,196 AI (39.9%)**. The policy
lives in `DatasetManager.DROID_LABEL_MAP`; override the class attribute to change
it.

The `Generator` column, recording which of 45 distinct models produced each AI
row, is preserved (lower-cased) as `ai_model` and powers the leave-one-model-out
test in Phase C.

## 3. Sampling caps — and how they interact

| Parameter | Value used | Role |
|---|---:|---|
| `--max-per-source` | **250,000** | Per-source ceiling applied at load time. Effectively uncapped for this build. |
| `--per-class` | **50,000** | Target rows per class after balancing. |

```bash
poetry run python scripts/build_training_set.py \
  --sources droidcollection \
  --per-class 50000 \
  --max-per-source 250000 \
  --output data/processed/train_droid_50k.parquet
```

> **The two flags are coupled, and the coupling is easy to miss.** The label map
> is applied **at load** (`dataset.py:634`), *before* `max_per_source` samples
> (`dataset.py:788`). The cap therefore draws from the 246,407 post-policy rows
> at DroidCollection's natural **60.1 / 39.9** class ratio — so a cap of `C`
> yields roughly `0.3985 × C` AI rows, and `balance()` clamps `--per-class` to
> whatever the scarcer class supplies.
>
> A cap of 100,000 yields only ~39,500 AI rows and silently produces a
> **39.5k/class** corpus against a 50k request. Reaching 50,000/class requires
> `--max-per-source ≥ 126,503`; **≥ 137,000** to still clear 50k/class after
> Phase B rejects (§6). `balance()` does print a warning when it clamps
> (`build_training_set.py:155`), but only after the run.
>
> Setting the cap at or above 246,407 removes the coupling entirely. Its original
> purpose — bounding memory — is not a concern for a single source: at a mean
> 1,099 characters per sample, all 246,407 rows are ~0.27 GB of strings.

## 4. Filtering and deduplication pipeline

The samples pass through four filters **in a fixed order**. The order is
load-bearing: conflict removal must precede deduplication, or the rows that
disagree would be silently collapsed.

| Step | Filter | Rows removed |
|---|---|---:|
| 4.1 | Python-only | 0 (already Python by construction) |
| 4.2 | Minimum length (< 10 tokens) | **1,544** |
| 4.3 | Label-conflict removal (all copies) | **0** |
| 4.4 | Exact-duplicate `source_code` | **475** |

**Usable unique samples: 244,388** — **147,308 human / 97,080 AI**.

Two of these numbers are much smaller than in the three-source build, and both
for the same reason. **Label conflicts are zero**: that pathology was purely an
AIGCodeSet↔CodeNet artifact (an LLM reproducing a human solution verbatim across
two corpora that disagreed about it). **Deduplication removes only 475 rows**
because the cross-source overlap that made it load-bearing — AIGCodeSet's human
half being drawn from CodeNet — no longer exists.

Both filters are retained regardless. They are cheap, and they are the guarantee
that makes source order irrelevant to ground truth if a second source is ever
reintroduced.

## 5. Balancing to 50/50

The balanced target is `min(--per-class, n_human, n_ai)`. With 97,080 AI rows
available and 50,000 requested, the AI class is **not** exhausted:

| Class | Rows | Share |
|---|---:|---:|
| Human (`label = 0`) | 50,000 | 50.0% |
| AI (`label = 1`) | 50,000 | 50.0% |
| **Total** | **100,000** | 100% |

Written to `data/processed/train_droid_50k.parquet`. The AI half spans **41
distinct generator models**.

**The ceiling for this corpus is 194,160 rows (97,080/class)**, set by AI supply.
The build is cap-limited by choice, not supply-limited — see §7.

## 6. Feature-extraction losses (Phase B)

Extraction is not lossless: a row is rejected when the source fails to parse or a
feature family fails. Of 100,000 rows, **94,108 were extracted and 5,892
rejected (5.89%)**, leaving:

| Class | Extracted | Rejected | Reject rate |
|---|---:|---:|---:|
| Human | 48,264 | 1,736 | **3.47%** |
| AI | 45,844 | 4,156 | **8.31%** |
| **Total** | **94,108** | **5,892** | **5.89%** |

`data/processed/train_features_checkpoint_droid_50k.parquet` retains all 100,000
rows with a `features_extracted` boolean, and is the only record of *which* rows
failed. Keep it alongside the extracted matrix.

### 6.1 Rejection is class-biased, and concentrated in base models

**AI code is rejected 2.4× more often than human code.** The cause is not diffuse
— it is concentrated in non-instruction-tuned generators:

| Generator | Rows | Rejected | Rate |
|---|---:|---:|---:|
| `qwen/qwen2.5-coder-7b` (**base**) | 1,459 | 487 | **33.4%** |
| `qwen/qwen2.5-coder-7b-instruct` | 2,014 | 58 | 2.9% |
| `meta-llama/llama-3.1-8b-instruct` | 1,499 | 69 | 4.6% |
| `qwen/qwen2.5-coder-1.5b-instruct` | 1,818 | 43 | 2.4% |
| `gpt-4o-mini` | 14,937 | 25 | 0.2% |

The base Qwen loses a third of its samples while its instruction-tuned sibling
loses 3%. Base models emit raw completions that trail off mid-statement or carry
prose alongside code, and neither parses.

**This is a selection bias and should be stated as one.** The AI class that
survives into training is the *parseable* subset of AI code. A claim of the form
"the detector achieves recall R on AI code" is properly read as "on AI code that
parses." Base-model output is under-represented in the trained corpus relative to
its true share, which also thins the held-out slice when a base model is chosen
as the LOMO target.

The bias is inherent to a pipeline that requires a valid AST, not a defect in
this build — the superseded three-source build lost ~7.0% of DroidCollection rows
the same way. It is recorded here so it is not rediscovered downstream.

### 6.2 What Phase C receives

Phase C rebalances the extracted matrix to the scarcer class: **45,844 per class,
91,688 rows**, discarding 2,420 surplus human rows.

All 41 generators survive extraction, and **24 retain ≥ 500 samples** — the
threshold at which a leave-one-model-out slice is meaningful. `gpt-4o-mini` is
the natural default holdout at **14,912** surviving rows.

## 7. Scaling headroom

The 100,000-row corpus is cap-limited, not supply-limited. AI is the binding
class:

| | Available (post-filter) |
|---|---:|
| Human | 147,308 |
| AI | **97,080** |
| **Balanced ceiling** | **194,160** (97,080/class) |

Raising `--per-class` toward 97,080 roughly doubles the corpus. Beyond that,
more AI samples require a corpus other than DroidCollection — which reintroduces
the pooling problem in §1.1 and would need the cross-corpus question settled
first.

**Cost scales linearly.** Every added sample is scored against the
350M-parameter CodeGen model in Phase B. This build's 100,000 rows took roughly
1.5–3 hours on a T4; the full ceiling would take proportionally longer.
Extraction checkpoints every 500 rows and resumes, so long sessions are safe.

> **Note the constraint flipped.** In the three-source build, CodeNet supplied
> the human class and DroidCollection's human half was starved to 3,832 rows by
> the 40,000 cap — making **human** the binding class at 3,700 rows after
> extraction. In a DroidCollection-only build, human is 1.5× more abundant than
> AI. Any intuition about corpus size carried over from the earlier build is
> inverted.

## 8. Superseded: the three-source build (22,000 rows)

Retained for provenance. The runs predating the switch were all trained on this
corpus (`features_sha256`
`f0dad5e1b9630ad3d2c64e18ed3f38734c9b1186fc90a8ce638137af365ef404`), and those
numbers are interpretable only against this construction. **Do not use it for new
work** — §1.1 explains why.

> **Removed runs.** The four pooled / no-CodeNet runs
> (`2026-07-17T13-16-24Z_gemini`, `2026-07-19T08-53-52Z_no-codenet`,
> `2026-07-19T09-40-53Z_no-codenet-gpt4omini`,
> `2026-07-19T10-04-00Z_no-codenet-codestral`) were deleted from `data/runs/` and
> pruned from `index.jsonl` once the single-corpus decision was final — their
> results were no longer needed. The construction narrative here is their
> remaining record. The droid-only (7,400-row) and transfer runs, which share the
> same `features_sha256`, still exist in `index.jsonl`.

**Sources and raw supply (≈ 493,990 Python samples):**

| Source | Origin | Raw Python content |
|---|---|---|
| AIGCodeSet | HF `basakdemirok/AIGCodeSet` | 7,583 (4,755 human + 2,828 AI) |
| CodeNet | IBM Project CodeNet, `Python800` | 240,000 (all human) |
| DroidCollection | HF `project-droid/DroidCollection` | 246,407 (after label mapping) |

AIGCodeSet's AI half was generated by CodeLlama 34B, Codestral 22B and Gemini 1.5
Flash; its human half is drawn from CodeNet, which is why cross-source dedup
mattered. AIGCodeSet was listed **first**, so it won deduplication collisions.
CodeNet is the `Python800` benchmark subset: 800 problems × 300 accepted
submissions.

**Caps:** `--max-per-source 40,000`, `--per-class 11,000`. DroidCollection
246,407 → 40,000; CodeNet 240,000 → 40,000; AIGCodeSet 7,583 taken in full.
**87,583 samples entered the filter pipeline.**

**Filter losses:** 19 too short · **210 label-conflict rows across 105 distinct
strings** · 119 exact duplicates.

**Unified filtered corpus: 87,235** — 68,529 human (78.6%) / 18,706 AI (21.4%);
by source DroidCollection 39,986 · CodeNet 39,914 · AIGCodeSet 7,335. The corpus
was human-heavy (≈ 3.7 : 1) because CodeNet is human-only.

**Final balanced corpus: 22,000** (11,000/class), spanning 44 generators:

| Source | Human | AI | Total |
|---|---:|---:|---:|
| CodeNet | 6,416 | 0 | 6,416 |
| DroidCollection | 3,832 | 9,430 | 13,262 |
| AIGCodeSet | 752 | 1,570 | 2,322 |
| **Total** | **11,000** | **11,000** | **22,000** |

After extraction: 20,756 rows (CodeNet 6,307 · DroidCollection 12,335 ·
AIGCodeSet 2,114). The droid-only experiments run against this matrix used only
its 12,335 DroidCollection rows, rebalanced to **3,700/class — 7,400 rows**.

## 9. Reproducibility

The procedure is deterministic under a single seed (`random_seed = 42`), which
governs the per-source subsampling, the per-class balancing draw, and the final
shuffle. Re-running `scripts/build_training_set.py` with the §3 arguments
reproduces the 100,000-row corpus exactly.

Phase B is deterministic given the same model, device, window and stride, but
`DELTX_DEVICE`, `DELTX_MAX_SEQUENCE_LENGTH` and `DELTX_SURPRISAL_STRIDE` change the
extracted values (the last two govern the sliding-window surprisal for files
longer than the window) —
pin them when reproducing a feature matrix. Phase C manifests record the
matrix's `sha256`, which is the authoritative identifier for any published
number; see [training.md](training.md#run-capture--making-results-citable).

## 10. Summary of key figures — current build

| Quantity | Value |
|---|---:|
| DroidCollection Python rows | 328,383 |
| Dropped by label policy (refined + adversarial) | 81,976 (25.0%) |
| Usable after label policy | 246,407 |
| Removed — too short (< 10 tokens) | 1,544 |
| Removed — label conflicts | 0 |
| Removed — exact duplicates | 475 |
| Unified filtered corpus | 244,388 (147,308 human / 97,080 AI) |
| Per-class target (`--per-class`) | 50,000 |
| **Final balanced corpus** | **100,000 (50,000 / 50,000)** |
| Rejected during feature extraction | 5,892 (5.89%) — human 3.47%, AI 8.31% |
| **Feature matrix** | **94,108 (48,264 human / 45,844 AI)** |
| Rebalanced for Phase C | 91,688 (45,844/class) |
| Distinct generators represented | 41 (24 with ≥ 500 surviving samples) |
| Balanced ceiling for this corpus | 194,160 (97,080/class) |

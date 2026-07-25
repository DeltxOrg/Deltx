# Feature-family ablation — Deltx AI-code detector

Corpus: `droidcollection`. 5 seeds, fixed hyperparameters across all arms.
Families are unequal in size (perplexity 6, stylometric 6, distribution 4); deltas are reported raw.

## Table 1 — Ablation arms

| Arm | #feat | AUROC | AUPRC | ΔAUROC vs full | 95% CI |
|---|---|---|---|---|---|
| `full_16` | 16 | 0.9538 ± 0.0017 | 0.9538 | — | — |
| `drop_perplexity` | 10 | 0.9255 ± 0.0020 | 0.9183 | -0.0283 | ±0.0005 |
| `drop_stylometric` | 10 | 0.8106 ± 0.0025 | 0.8312 | -0.1432 | ±0.0009 |
| `drop_distribution` | 12 | 0.9469 ± 0.0017 | 0.9466 | -0.0069 | ±0.0004 |
| `only_perplexity` | 6 | 0.7690 ± 0.0026 | 0.7898 | -0.1848 | ±0.0012 |
| `only_stylometric` | 6 | 0.9059 ± 0.0025 | 0.8931 | -0.0479 | ±0.0008 |
| `only_distribution` | 4 | 0.6654 ± 0.0009 | 0.6828 | -0.2884 | ±0.0015 |

Read `drop_X` as the cost of removing family X (larger negative Δ = more
necessary). Read `only_X` as how far family X gets on its own
(less negative = more sufficient). A family can be sufficient but not
necessary when families are redundant — compare the two columns.

## Table 2 — Operating point at 5% FPR (threshold chosen on validation)

| Arm | threshold | precision | recall |
|---|---|---|---|
| `full_16` | 0.777 | 0.9387 | 0.7655 |
| `drop_perplexity` | 0.825 | 0.9243 | 0.6308 |
| `drop_stylometric` | 0.726 | 0.8910 | 0.4306 |
| `drop_distribution` | 0.787 | 0.9346 | 0.7369 |
| `only_perplexity` | 0.745 | 0.8740 | 0.3586 |
| `only_stylometric` | 0.834 | 0.9130 | 0.5401 |
| `only_distribution` | 0.729 | 0.8149 | 0.2223 |

## Table 3 — Threshold sensitivity (full 16-feature model)

| threshold | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|
| 0.05 | 0.7489 | 0.6678 | 0.9908 | 0.7978 | 0.4930 |
| 0.10 | 0.8055 | 0.7262 | 0.9808 | 0.8345 | 0.3699 |
| 0.15 | 0.8344 | 0.7625 | 0.9714 | 0.8544 | 0.3025 |
| 0.20 | 0.8526 | 0.7890 | 0.9625 | 0.8672 | 0.2574 |
| 0.25 | 0.8653 | 0.8108 | 0.9529 | 0.8761 | 0.2223 |
| 0.30 | 0.8729 | 0.8275 | 0.9421 | 0.8811 | 0.1964 |
| 0.35 | 0.8780 | 0.8423 | 0.9301 | 0.8840 | 0.1742 |
| 0.40 | 0.8821 | 0.8565 | 0.9181 | 0.8862 | 0.1539 |
| 0.45 | 0.8842 | 0.8687 | 0.9051 | 0.8865 | 0.1368 |
| 0.50 | 0.8847 | 0.8805 | 0.8903 | 0.8854 | 0.1208 |
| 0.55 | 0.8844 | 0.8917 | 0.8751 | 0.8833 | 0.1062 |
| 0.60 | 0.8824 | 0.9022 | 0.8578 | 0.8795 | 0.0930 |
| 0.65 | 0.8786 | 0.9124 | 0.8376 | 0.8734 | 0.0804 |
| 0.70 | 0.8733 | 0.9229 | 0.8146 | 0.8654 | 0.0681 |
| 0.75 | 0.8642 | 0.9333 | 0.7844 | 0.8524 | 0.0561 |
| 0.80 | 0.8509 | 0.9425 | 0.7473 | 0.8337 | 0.0456 |
| 0.85 | 0.8313 | 0.9538 | 0.6964 | 0.8050 | 0.0338 |
| 0.90 | 0.7974 | 0.9667 | 0.6160 | 0.7525 | 0.0212 |
| 0.95 | 0.7294 | 0.9813 | 0.4678 | 0.6335 | 0.0089 |

The default 0.5 is a reporting convention, not a tuned choice. Deltx
consumes `ai_confidence` as a continuous signal downstream, so the
threshold only affects the reported confusion matrix.

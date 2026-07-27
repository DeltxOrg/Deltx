# Feature-family ablation — Deltx AI-code detector

Corpus: `aigcodeset`. 5 seeds, fixed hyperparameters across all arms.
Families are unequal in size (perplexity 6, stylometric 6, distribution 4); deltas are reported raw.

## Table 1 — Ablation arms

| Arm | #feat | AUROC | AUPRC | ΔAUROC vs full | 95% CI |
|---|---|---|---|---|---|
| `full_16` | 16 | 0.6382 ± 0.0147 | 0.6774 | — | — |
| `drop_perplexity` | 10 | 0.5774 ± 0.0108 | 0.6119 | -0.0608 | ±0.0070 |
| `drop_stylometric` | 10 | 0.5926 ± 0.0156 | 0.6141 | -0.0456 | ±0.0103 |
| `drop_distribution` | 12 | 0.6305 ± 0.0187 | 0.6712 | -0.0078 | ±0.0044 |
| `only_perplexity` | 6 | 0.5684 ± 0.0155 | 0.5853 | -0.0698 | ±0.0113 |
| `only_stylometric` | 6 | 0.5621 ± 0.0160 | 0.6010 | -0.0762 | ±0.0077 |
| `only_distribution` | 4 | 0.5000 ± 0.0095 | 0.5057 | -0.1382 | ±0.0193 |

Read `drop_X` as the cost of removing family X (larger negative Δ = more
necessary). Read `only_X` as how far family X gets on its own
(less negative = more sufficient). A family can be sufficient but not
necessary when families are redundant — compare the two columns.

## Table 2 — Operating point at 5% FPR (threshold chosen on validation)

| Arm | threshold | precision | recall |
|---|---|---|---|
| `full_16` | 0.867 | 0.8480 | 0.1969 |
| `drop_perplexity` | 0.852 | 0.7770 | 0.1304 |
| `drop_stylometric` | 0.858 | 0.7324 | 0.1173 |
| `drop_distribution` | 0.858 | 0.8271 | 0.2027 |
| `only_perplexity` | 0.830 | 0.7035 | 0.1120 |
| `only_stylometric` | 0.809 | 0.7250 | 0.1442 |
| `only_distribution` | 0.811 | 0.5536 | 0.0482 |

## Table 3 — Threshold sensitivity (full 16-feature model)

| threshold | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|
| 0.05 | 0.5028 | 0.5013 | 0.9963 | 0.6670 | 0.9906 |
| 0.10 | 0.5054 | 0.5028 | 0.9698 | 0.6622 | 0.9587 |
| 0.15 | 0.5197 | 0.5107 | 0.9338 | 0.6603 | 0.8942 |
| 0.20 | 0.5289 | 0.5168 | 0.8823 | 0.6518 | 0.8243 |
| 0.25 | 0.5465 | 0.5296 | 0.8296 | 0.6464 | 0.7365 |
| 0.30 | 0.5549 | 0.5383 | 0.7707 | 0.6337 | 0.6610 |
| 0.35 | 0.5620 | 0.5476 | 0.7131 | 0.6193 | 0.5891 |
| 0.40 | 0.5773 | 0.5655 | 0.6669 | 0.6120 | 0.5123 |
| 0.45 | 0.5904 | 0.5849 | 0.6220 | 0.6028 | 0.4412 |
| 0.50 | 0.6006 | 0.6056 | 0.5770 | 0.5907 | 0.3758 |
| 0.55 | 0.6049 | 0.6245 | 0.5259 | 0.5705 | 0.3162 |
| 0.60 | 0.6102 | 0.6480 | 0.4822 | 0.5526 | 0.2619 |
| 0.65 | 0.6112 | 0.6716 | 0.4340 | 0.5266 | 0.2116 |
| 0.70 | 0.6072 | 0.6969 | 0.3788 | 0.4900 | 0.1646 |
| 0.75 | 0.6047 | 0.7342 | 0.3269 | 0.4517 | 0.1177 |
| 0.80 | 0.5978 | 0.7666 | 0.2799 | 0.4094 | 0.0846 |
| 0.85 | 0.5855 | 0.8208 | 0.2190 | 0.3450 | 0.0482 |
| 0.90 | 0.5661 | 0.8844 | 0.1520 | 0.2591 | 0.0200 |
| 0.95 | 0.5385 | 0.9251 | 0.0838 | 0.1536 | 0.0069 |

The default 0.5 is a reporting convention, not a tuned choice. Deltx
consumes `ai_confidence` as a continuous signal downstream, so the
threshold only affects the reported confusion matrix.

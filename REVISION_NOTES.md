# Revision notes

Following peer review, a self-audit of the training code identified reporting defects
in the previously published results. This file records what was found, what has been
withdrawn, and what is being done about it. Items are listed in the order they affect
the manuscript.

---

## R1 — Foundation model comparison withdrawn

The RETFound comparison and the main-branch results were computed on **different test
partitions**.

| Class | RETFound run | Main branch |
|---|---|---|
| CNV | 3,746 | 3,960 |
| DME | 1,161 | 1,101 |
| DRUSEN | 887 | 1,086 |
| NORMAL | 5,139 | 1,786 |
| **Total** | **10,933** | **7,933** |

Support counts are from the per-seed `classification_report` entries in
`retfound/results_full_finetune_3seed.json`, and are identical across seeds 42, 123
and 999.

NORMAL — the easiest class — makes up 47% of the RETFound partition and 22.5% of the
main-branch partition, so the two accuracies are not comparable in either direction.
The published claim of parity with RETFound is therefore unsupported and has been
withdrawn.

A third partition (61,212 / 7,651 / 7,652) exists as `data_splits/*.npy` on the
HuggingFace weights repository, giving three incompatible data pipelines across the
project.

**Action:** standardise on one split; re-evaluate both models on it; republish.

**Also disclosed:** the RETFound run substituted seed 999 for seed 2024 after float16
AMP instability. This narrows its reported spread and must be stated wherever that
spread is cited.

---

## R2 — Expected Calibration Error withdrawn

The ECE routine iterated its bin edges with

```python
for lo, hi in zip(*[iter(np.linspace(0, 1, n_bins + 1))]*2):
```

which chunks the edge array into consecutive non-overlapping pairs rather than sliding
across it. At `n_bins=15` this evaluates 8 intervals and skips 7, so any prediction
whose confidence falls in a gap is excluded, and the bin weights no longer sum to 1.
The result is a systematic underestimate.

Simulated on a test set of the same size and confidence profile: 38.3% of samples
excluded, ECE understated by a factor of 1.42.

The defect is present in **both** copies of the function — Phase 4 (Cell 4.7) and
Phase 6 (Cell 6.2) — so all five seeds are affected. The published figure of
0.0024 ± 0.0005 should not be cited.

**Action:** corrected implementation; rerun Phase 6; republish the real value.

---

## R3 — Ablation table withdrawn

The published ablation described an architectural progression — frozen backbone,
then block-6 fine-tuning, then the transformer, then the XGBoost head. Producing those
four rows requires training four models, and that code is not in this repository.

The ablation the notebook actually runs (Phase 5, Section 4) compares **classifier
heads on a single frozen feature extractor**: logistic regression, calibrated linear
SVM, a reduced-capacity XGBoost, a backbone-only XGBoost, and the final hybrid. That is
a different experiment and does not substitute for the progression.

Separately, the variant labelled `XGBoost (no CutMix)` does not disable CutMix — it
reduces `n_estimators` to 100 and `max_depth` to 6, operating on features extracted
from the CutMix-trained model. The label is incorrect.

**Action:** implement the progression properly, or publish the head-comparison
ablation under its correct description.

---

## R4 — McNemar comparator now stated

The published `p < 0.0001` came from a McNemar test of the **XGBoost head against
logistic regression on identical features** (Phase 6). The results table gave no
comparator, which invited the reading that it referred to the baseline CNN or to
RETFound. On 7,933 samples that particular test is significant essentially by
construction.

**Action:** state the comparator wherever the p-value appears, and run the test against
a meaningful comparison once R1 places both models on one split.

---

## R5 — Validation split too small for its jobs

`val_gen` points at the dataset's official `val/` directory, which contains 32 images
(8 per class). That split currently drives Optuna hyperparameter search, early
stopping, checkpoint selection, LR scheduling, temperature scaling, and the
Mahalanobis OOD threshold at `np.percentile(val_distances, 97)`.

A 97th percentile estimated from 32 points is the maximum of 32 points, so the OOD
threshold and the temperature are both fit on a sample too small for the quantity
being estimated.

**Action:** carve a patient-disjoint, class-balanced validation split from `train/`
using the Kermany patient IDs encoded in each filename; refit temperature and the OOD
threshold on it; verify and state patient-independence against the test split.

---

## R6 — Hyperparameter variance not captured

Optuna was run once on seed 42 and the selected hyperparameters reused for all five
seeds. The reported ±0.27% therefore reflects seed variance only and excludes
hyperparameter-selection variance. Stated as a limitation.

---

## R7 — Repository hygiene

- CI badge pointed at a workflow path that does not exist in this repository (404).
- Setup instructions referenced `reproduce/environment.yml`, which does not exist.
- Repository structure listed `retfound/`, which lives on the `retfound-finetune`
  branch, not `main`.
- The citation section carried a stale note reading "once the arXiv preprint is
  submitted", which had not been removed after
  [arXiv:2607.09809](https://arxiv.org/abs/2607.09809) went live on 9 July 2026. The
  BibTeX entry also gave a working title that does not match the posted preprint.
  Both now match the arXiv record, and the preprint and software archive are cited
  separately.
- The Phase 6 model card carried `[Your Name]` / `[Your University]` placeholders and
  a stale date.

All corrected or removed.

---

## What is unaffected

The following were specifically checked and found sound:

- The XGBoost head is fit on training features only. There is no `eval_set` on the
  test split and no early stopping against it.
- Temperature scaling and the OOD threshold are fit on the validation split, not the
  test split. (Their problem is the split's size — R5 — not leakage.)
- The five seeds genuinely retrain end to end: Phase A, Phase B, feature extraction and
  the XGBoost head are all rerun per seed. The reported seed variance is real.
- Deployment preprocessing in `app.py` matches the training augmentation pipeline
  exactly; there is no train/serve skew.
- Bootstrap confidence intervals are computed for AUC.

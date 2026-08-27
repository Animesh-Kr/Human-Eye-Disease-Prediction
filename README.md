# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**
MSc Advanced Computer Science — Newcastle University (2025–26)

[![arXiv](https://img.shields.io/badge/arXiv-2607.09809-B31B1B.svg)](https://arxiv.org/abs/2607.09809)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19224303.svg)](https://doi.org/10.5281/zenodo.19224303)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Gradio%20Pipeline-yellow)](https://huggingface.co/spaces/animeshakr/oct-complete-pipeline)
[![Dashboard](https://img.shields.io/badge/🤗%20HuggingFace-Streamlit%20Dashboard-orange)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![API](https://img.shields.io/badge/🤗%20HuggingFace-REST%20API-green)](https://huggingface.co/spaces/animeshakr/oct-retinal-api)
[![Model Weights](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--0608--7004-brightgreen)](https://orcid.org/0009-0003-0608-7004)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Status

This work is **under revision following peer review**. A self-audit of the training
code found reporting defects that are being corrected before resubmission. Affected
claims are marked ⚠️ throughout and are listed in [`REVISION_NOTES.md`](REVISION_NOTES.md).

The preprint [arXiv:2607.09809](https://arxiv.org/abs/2607.09809) (submitted 9 July
2026) predates this audit and still contains the affected claims. **A corrected v2 is
in preparation.** Until it is posted, prefer the figures in this README over those in
v1.

---

## Overview

A hybrid CNN-Transformer framework for four-class retinal OCT classification. The
pipeline covers raw B-scan preprocessing through to edge-optimised inference, with
explicit attention to calibration, out-of-distribution rejection, and uncertainty —
mechanisms absent from most published OCT classifiers.

The model achieves **95.43% ± 0.27% accuracy** across five independent random seeds,
each retrained end to end. The edge-optimised ONNX node reduces the 2.07 GB Keras
model to 237 MB and runs at ~62.9 ms per scan on a standard CPU — no GPU required.

---

## Results — 5-Seed Statistical Validation

Test set: Kermany official `test/` split, **n = 7,933** (CNV 3,960 · DME 1,101 ·
DRUSEN 1,086 · NORMAL 1,786). Each seed retrains the full pipeline: Phase A → Phase B
→ feature extraction → XGBoost head.

| Metric | Mean ± Std |
|---|---|
| Accuracy | **95.43% ± 0.27%** |
| Macro AUC-ROC | **0.9941 ± 0.0006** |
| Macro F1 | 0.9244 ± 0.0047 |
| Drusen F1 (minority class) | 0.8436 ± 0.0096 |
| ECE (calibrated) | ⚠️ recomputation pending — see below |

⚠️ **ECE withdrawn.** The expected-calibration-error routine binned its confidence
axis incorrectly, evaluating 8 of 15 bins and dropping the samples that fell in the
gaps. The previously published figure (0.0024 ± 0.0005) understates the true value and
should not be cited. Corrected implementation and the real number will land with the
revision.

Hyperparameters were tuned once (Optuna, seed 42) and reused across all five seeds, so
the reported ±0.27% reflects seed variance only and excludes hyperparameter-selection
variance.

---

## Foundation Model Comparison — RETFound (ViT-L/16 MAE)

**Correction (supersedes an earlier note in this README).** An earlier revision of this
file withdrew this comparison on the belief that the two models had been scored on
different test partitions. That was wrong. Both were evaluated on the **same 10,933-image
test set**: `assets/clinical_metrics.csv` gives per-class support of CNV 3,746 /
DME 1,161 / DRUSEN 887 / NORMAL 5,139, identical to the supports in
`retfound/results_full_finetune_3seed.json`, and its implied accuracy (10,484/10,933
= 0.9589) matches `assets/ablation_results.csv` exactly. The comparison stands.

| Method | Accuracy | AUC-ROC | Macro F1 | Drusen F1 | Seeds |
|---|---|---|---|---|---|
| **Ours (EffNetV2L + 4×MHA + XGBoost)** | **95.43% ± 0.27%** | **0.9941 ± 0.0006** | **0.9244 ± 0.0047** | **0.8436 ± 0.0096** | **5** |
| RETFound full fine-tuning (303M params) | 95.14% ± 1.37% | 0.9922 ± 0.0041 | 0.9189 ± 0.0214 | 0.8192 ± 0.0420 | 3 |
| RETFound linear probe (frozen ViT-L/16) | 75.72% ± 0.67% | 0.9103 ± 0.0002 | 0.6590 ± 0.0052 | 0.3829 ± 0.0009 | 3 |

Two caveats that belong with the table: the RETFound run substituted seed 999 for seed
2024 after float16 AMP instability, which narrows its reported spread; and RETFound's
best seed (96.33%) exceeds our mean, so the honest claim is parity, not superiority.

Experiment code and raw per-seed results remain available for inspection in the
[`retfound-finetune`](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction/tree/retfound-finetune)
branch and on [HuggingFace](https://huggingface.co/animeshakr/oct-retinal-weights/tree/main/retfound).
Note that the RETFound run substituted seed 999 for seed 2024 after float16 AMP
instability; that substitution narrows its reported spread.

---

## Ablation Study

The architectural progression previously published here (frozen backbone → +block6 →
+transformer → +XGBoost) is withdrawn: no code in this repository produces it.

This is the ablation the notebook actually runs, from `assets/ablation_results.csv` —
five classifier heads on one frozen feature extractor, all on the 10,933-image test set:

| Head | Accuracy | Macro F1 | Drusen F1 | Macro AUC |
|---|---|---|---|---|
| Logistic Regression | 0.9639 | 0.9374 | 0.8645 | 0.9949 |
| SVM (Linear, calibrated) | 0.9654 | 0.9395 | 0.8686 | 0.9932 |
| XGBoost (reduced capacity) | 0.9632 | 0.9364 | 0.8612 | 0.9948 |
| **XGBoost (backbone-only)** | **0.9659** | **0.9408** | 0.8679 | **0.9955** |
| XGBoost Hybrid (final) | 0.9589 | 0.9316 | 0.8553 | 0.9947 |

⚠️ **The XGBoost hybrid head is the weakest of the five.** Logistic regression beats it
by 0.5 points of accuracy, and a backbone-only XGBoost — which does not use the
transformer features at all — beats it by 0.7 points and leads on every metric except
Drusen F1. On this evidence the transformer-plus-XGBoost head is not justified as the
contribution, and the architecture's value has to be re-argued or the head replaced.

The variant labelled `XGBoost (no CutMix)` in the raw CSV is renamed above: it does not
disable CutMix, it reduces `n_estimators` to 100 and `max_depth` to 6 on features from
the CutMix-trained model.

---

## Architecture

```
OCT Scan (224×224×3)
    → EfficientNetV2L backbone (118.5M params, ImageNet-21k pretrained)
       Blocks 1–5: frozen  |  Block 6+: fine-tuned
    → Patch reshape: 7×7×1280 → 49 tokens
    → Linear projection: 1280 → 256-d
    → Learnable Positional Encoding
    → 4× Multi-Head Attention (16 heads, key_dim=16)
    → GlobalAvgPool1D → 256-d feature vector
    → XGBoost hybrid head (300 trees, max_depth=4)
    → Temperature scaling
    → CNV / DME / DRUSEN / NORMAL
```

Head count, projection width, dropout, focal gamma and XGBoost depth are selected by
Optuna over `n_heads ∈ {8, 16}`, `proj_dim ∈ {256, 512}`, `xgb_depth ∈ [4, 8]`. The
configuration actually deployed is the one recorded in `best_hparams.pkl`, which the
diagram above reflects: `n_heads=16`, `proj_dim=256` (hence `key_dim=16`),
`xgb_depth=4`, `lr=1.59e-4`, `dropout=0.383`, `focal_gamma=1.364`.

---

## Clinical Safety Features

| Feature | Method | Purpose |
|---|---|---|
| OOD Detection | Mahalanobis distance, 97th-percentile threshold | Rejects non-retinal or corrupted scans |
| Uncertainty | MC Dropout (20 passes), σ > 0.15 | Flags low-confidence predictions |
| Calibration | Temperature scaling | Corrects systematic softmax overconfidence |
| Explainability | Grad-CAM + SHAP | Spatial + feature attribution |

⚠️ **Calibration caveat.** Both the temperature and the OOD threshold are currently fit
on the dataset's official `val/` directory, which holds 32 images (8 per class) in the
standard OCT2017 release. A 97th
percentile estimated from 32 points is not a meaningful quantile. A patient-disjoint
validation split is being carved from `train/` and both quantities refit on it.

**Failure case caught by the safety layer:** a NORMAL scan was classified as DRUSEN at
66.85% confidence. MC Dropout σ exceeded 0.15, routing the scan to specialist review
rather than silently returning an incorrect prediction.

---

## Edge Deployment

| Format | Size | Reduction |
|---|---|---|
| Keras `.keras` (master) | 2,070 MB | baseline |
| **ONNX FP32 (edge node)** | **237 MB** | **~88% smaller** |

**Latency benchmark (Intel CPU, batch size 1, 50 runs):**

| Scan | Prediction | Confidence | Latency |
|---|---|---|---|
| DRUSEN | DRUSEN ✅ | 92.72% | 61.96 ms |
| DME | DME ✅ | 81.20% | 62.60 ms |
| CNV | CNV ✅ | 91.46% | 64.31 ms |
| NORMAL | DRUSEN ⚠️ | 66.85% | 62.82 ms |
| **Global Average** | — | — | **~62.9 ms** |

---

## Live Deployments

| Space | Description | Link |
|---|---|---|
| 🟣 Gradio Complete Pipeline | Classification + segmentation routing | [oct-complete-pipeline](https://huggingface.co/spaces/animeshakr/oct-complete-pipeline) |
| 🔵 Streamlit Dashboard | Full diagnostic dashboard with Grad-CAM | [oct-retinal-ai](https://huggingface.co/spaces/animeshakr/oct-retinal-ai) |
| 🟢 FastAPI REST | JSON inference endpoint | [oct-retinal-api](https://huggingface.co/spaces/animeshakr/oct-retinal-api) |
| 📦 Model Weights | ONNX + Keras + safety components | [oct-retinal-weights](https://huggingface.co/animeshakr/oct-retinal-weights) |

---

## Repository Structure

```
├── deployment_cloud/           # FastAPI Docker deployment
├── edge_inference/             # ONNX edge node (237 MB, ~62.9ms CPU)
├── Human-Eye.ipynb             # Full training pipeline — Phases 1–6
├── app.py                      # Streamlit dashboard
├── quantise_benchmark.py       # ONNX export + latency benchmark
└── assets/                     # Attention maps, confusion matrix, plots
```

The RETFound comparison lives on the
[`retfound-finetune`](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction/tree/retfound-finetune)
branch, not on `main`.

---

## Setup

```bash
pip install -r requirements_local.txt
```

Then fetch the weights and safety components:

```python
from huggingface_hub import hf_hub_download
import os
os.makedirs('models', exist_ok=True)
for f in ['Final_CNN_Transformer.keras', 'Final_XGBoost_Hybrid.json',
          'ood_train_mean.npy', 'ood_cov_inv.npy',
          'ood_threshold.npy', 'temperature.npy']:
    hf_hub_download(repo_id='animeshakr/oct-retinal-weights',
                    filename=f, local_dir='models/')
```

---

## Dataset

Kermany et al. (Cell 2018) — 84,495 OCT B-scans, 4 classes.

| Class | Val | Test | Clinical Significance |
|---|---|---|---|
| CNV | 8 | 3,746 | Wet AMD — urgent anti-VEGF |
| DME | 8 | 1,161 | Diabetic macular oedema |
| DRUSEN | 8 | 887 | Early AMD biomarker |
| NORMAL | 8 | 5,139 | Healthy retina |
| **Total** | **32** | **10,933** | |

⚠️ Test counts are derived from `assets/clinical_metrics.csv`, i.e. from the split the
pipeline actually evaluated. An earlier version of this table reported the published
Kermany OCT2017 split (3,960 / 1,101 / 1,086 / 1,786 = 7,933) rather than the contents
of the directories this code downloads. Train counts are being recounted the same way.

The official `val/` split holds 32 images total. It is currently used for model
selection, temperature scaling and the OOD threshold; see the calibration caveat
above. A patient-disjoint split carved from `train/` replaces it in the revision.

---

## Comparison to Published Work

| Study | Backbone | Accuracy |
|---|---|---|
| Kermany et al. (Cell 2018) | VGG-16 | 96.6% |
| Fang et al. (2019) | Multi-scale CNN | 97.4% |
| Li et al. (2021) | Vision Transformer | 98.1% |
| **This work** | EfficientNetV2L + 4× MHA + XGBoost | **95.43% ± 0.27%** |

Accuracies are **not directly comparable** — these studies use different test
protocols, and only this work reports multi-seed variance. The contribution here is
not peak accuracy but the calibration, OOD-rejection and uncertainty layer, together
with a 237 MB CPU-only deployment node.

---

## Citation

Preprint:

```bibtex
@misc{kumar2026oct,
  author        = {Kumar, Animesh},
  title         = {Calibrated Hybrid {CNN}-Transformer for Retinal {OCT} Classification},
  year          = {2026},
  eprint        = {2607.09809},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  doi           = {10.48550/arXiv.2607.09809},
  url           = {https://arxiv.org/abs/2607.09809}
}
```

Software archive:

```bibtex
@software{kumar2026octcode,
  author  = {Kumar, Animesh},
  title   = {OCT Retinal AI --- Hybrid CNN-Transformer Classification Pipeline},
  year    = {2026},
  doi     = {10.5281/zenodo.19224303},
  url     = {https://doi.org/10.5281/zenodo.19224303},
  note    = {Independent research, Newcastle University MSc Advanced
             Computer Science 2025--26. Zenodo software archive v1.0.0.}
}
```

---

**Author:** Animesh Kumar · MSc Advanced Computer Science · Newcastle University (2025–26)
**ORCID:** [0009-0003-0608-7004](https://orcid.org/0009-0003-0608-7004)
**License:** MIT

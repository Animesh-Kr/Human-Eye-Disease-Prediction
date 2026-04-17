# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**  
MSc Advanced Computer Science — Newcastle University (2025–26)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19224303.svg)](https://doi.org/10.5281/zenodo.19224303)
[![CI](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction/actions/workflows/model_tests.yml/badge.svg?branch=retfound-finetune)](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction/actions/workflows/model_tests.yml)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Gradio%20Pipeline-yellow)](https://huggingface.co/spaces/animeshakr/oct-complete-pipeline)
[![Dashboard](https://img.shields.io/badge/🤗%20HuggingFace-Streamlit%20Dashboard-orange)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![API](https://img.shields.io/badge/🤗%20HuggingFace-REST%20API-green)](https://huggingface.co/spaces/animeshakr/oct-retinal-api)
[![Model Weights](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--0608--7004-brightgreen)](https://orcid.org/0009-0003-0608-7004)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This repository contains the full implementation of a hybrid CNN-Transformer framework for four-class retinal OCT classification. The pipeline covers raw B-scan preprocessing through to edge-optimised inference, with a focus on the clinical safety mechanisms absent from most published OCT models.

The master model achieves 95.43% accuracy across five independent random seeds. The edge-optimised ONNX node reduces the 2.07 GB Keras model to 237 MB and runs at ~62.9 ms per scan on a standard CPU — no GPU required.

---

## Results — 5-Seed Statistical Validation

| Metric | Mean ± Std |
|---|---|
| Accuracy | 95.43% ± 0.27% |
| Macro AUC-ROC | 0.9941 ± 0.0006 |
| Macro F1 | 0.9244 ± 0.0047 |
| Drusen F1 (minority class) | 0.8436 ± 0.0096 |
| ECE (calibrated) | 0.0024 ± 0.0005 |
| McNemar p-value | < 0.0001 (all 5 seeds) |

---

## Foundation Model Comparison — RETFound (ViT-L/16 MAE)

To contextualise our results, we fine-tuned [RETFound](https://huggingface.co/YukunZhou/RETFound_mae_meh) — a 303M-parameter ViT-L/16 foundation model pretrained on 1.6M retinal images from Moorfields Eye Hospital — on the same dataset and evaluation protocol. Full experiment code and results are in the [`retfound-finetune`](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction/tree/retfound-finetune) branch.

| Method | Accuracy | AUC-ROC | Macro F1 | Drusen F1 | Seeds |
|---|---|---|---|---|---|
| Baseline CNN (EffNetV2L + Dense) | 89.12% | 0.9410 | 0.8642 | — | 1 |
| **Ours (EffNetV2L + 4×MHA + XGBoost)** | **95.43% ± 0.27%** | **0.9941 ± 0.0006** | **0.9244 ± 0.0047** | **0.8436 ± 0.0096** | **5** |
| RETFound linear probe (frozen ViT-L/16) | 75.72% ± 0.67% | 0.9103 ± 0.0002 | 0.6590 ± 0.0052 | 0.3829 ± 0.0009 | 3 |
| RETFound full fine-tuning (303M params) | 95.14% ± 1.37% | 0.9922 ± 0.0041 | 0.9189 ± 0.0214 | 0.8192 ± 0.0420 | 3 |

**Key findings:**
- Our hybrid architecture matches a domain-specific 303M-parameter foundation model while being **5× more stable** across seeds (±0.27% vs ±1.37%)
- The XGBoost head outperforms the RETFound linear classification head on the minority Drusen class (F1 0.84 vs 0.82)
- RETFound linear probe (75.72%) confirms that the modality gap between fundus photographs and OCT B-scans requires full domain adaptation

---

## Ablation Study

| Variant | Accuracy | AUC-ROC |
|---|---|---|
| EfficientNetV2L (frozen) + Dense | 89.12% | 0.9410 |
| + Fine-tuned Block 6+ | 92.45% | 0.9755 |
| + Transformer (4× MHA) | 94.10% | 0.9880 |
| + XGBoost head (full model) | **95.43%** | **0.9941** |

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
    → Temperature scaling (T≈1.05)
    → CNV / DME / DRUSEN / NORMAL
```

---

## Clinical Safety Features

| Feature | Method | Threshold | Purpose |
|---|---|---|---|
| OOD Detection | Mahalanobis distance | 97th percentile | Rejects non-retinal or corrupted scans |
| Uncertainty | MC Dropout (20 passes) | σ > 0.15 | Flags low-confidence predictions |
| Calibration | Temperature scaling | T ≈ 1.05 | Corrects systematic softmax overconfidence |
| Explainability | Grad-CAM + SHAP | — | Spatial + feature attribution |

**Concrete failure case:** A NORMAL scan was classified as DRUSEN at 66.85% confidence. MC Dropout σ exceeded 0.15, routing the scan to specialist review instead of silently returning a wrong prediction.

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
├── retfound/                   # RETFound comparison experiment (retfound-finetune branch)
├── deployment_cloud/           # FastAPI Docker deployment
├── edge_inference/             # ONNX edge node (237 MB, ~62.9ms CPU)
├── .github/workflows/          # CI/CD — ONNX validation + API health check
├── Human-Eye.ipynb             # Full training pipeline — Phases 1–6
├── app.py                      # Streamlit dashboard
└── assets/                     # Attention maps, confusion matrix, ablation plots
```

---

## Setup

```bash
conda env create -f reproduce/environment.yml
conda activate oct_retinal
```

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

| Class | Train | Test | Clinical Significance |
|---|---|---|---|
| CNV | 37,206 | 3,960 | Wet AMD — urgent anti-VEGF |
| DME | 11,349 | 1,101 | Diabetic macular oedema |
| DRUSEN | 8,617 | 1,086 | Early AMD biomarker |
| NORMAL | 26,315 | 1,786 | Healthy retina |

---

## Citation

```bibtex
@article{kumar2026oct,
  author  = {Kumar, Animesh A.},
  title   = {A Hybrid {CNN}-Transformer Framework for Retinal {OCT}
             Classification with Integrated Clinical Safety Mechanisms},
  year    = {2026},
  doi     = {10.5281/zenodo.19224303},
  note    = {Zenodo software archive, v1.0.0}
}
```

---

**Author:** Animesh Kumar — MSc Advanced Computer Science, Newcastle University (2025–26)  
**License:** MIT

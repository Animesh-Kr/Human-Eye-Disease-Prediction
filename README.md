# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**  
MSc Advanced Computer Science — Newcastle University (2025–26)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19224303.svg)](https://doi.org/10.5281/zenodo.19224303)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![Model Weights](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--0608--7004-brightgreen)](https://orcid.org/0009-0003-0608-7004)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This repository contains the full implementation of a hybrid CNN-Transformer framework for four-class retinal OCT classification. The pipeline covers everything from raw B-scan preprocessing through to edge-optimised inference, with a particular focus on the clinical safety mechanisms that are absent from most published OCT models.

The master model runs at 95.43% accuracy across five independent random seeds. The edge-optimised ONNX node reduces the 2.07 GB Keras model to 237 MB and achieves ~62.9 ms per scan on a standard CPU — without a GPU dependency.

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

## Comparison with Published Methods

| Method | Backbone | Accuracy | Macro AUC | ECE | Safety |
|---|---|---|---|---|---|
| Kermany et al. 2018 | InceptionV3 | 96.6% | — | — | None |
| He et al. 2019 | DenseNet/ResNet | 93.2% | — | — | None |
| Li et al. 2021 | Attention DenseNet | 91.7% | 0.97 | — | None |
| **Ours** | **EfficientNetV2L + 4×MHA + XGBoost** | **95.43%** | **0.9941** | **0.0024** | **OOD + Uncertainty + Calibration** |

> The ~1% accuracy gap versus the Kermany InceptionV3 baseline is an intentional trade-off. The gain is clinical-grade calibration (ECE 0.0024, a 12-fold improvement) and three integrated safety mechanisms that none of the prior methods include.

---

## Baseline Comparison

| Model | Accuracy | Macro AUC-ROC | Macro F1 | ECE (Calibrated) |
|---|---|---|---|---|
| Baseline CNN (EffNetV2L + Dense) | 89.12% | 0.9410 | 0.8642 | 0.0203 |
| **Hybrid CNN-Transformer-XGBoost** | **95.43%** | **0.9941** | **0.9244** | **0.0024** |

---

## Ablation Study

| Variant | Accuracy | AUC-ROC |
|---|---|---|
| EfficientNetV2L (frozen) + Dense | 89.12% | 0.9410 |
| + Fine-tuned Block 6+ | 92.45% | 0.9755 |
| + Transformer (4× MHA) | 94.10% | 0.9880 |
| + XGBoost head (full model) | **95.43%** | **0.9941** |

Each component adds a measurable increment — no single stage accounts for the full gap over the baseline.

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

The safety envelope was motivated by a concrete failure case observed during edge benchmarking: a NORMAL scan was classified as DRUSEN at 66.85% confidence. On its own, that prediction would go unchallenged. The three mechanisms below exist precisely to catch this kind of case before it reaches a clinician.

| Feature | Method | Threshold | Purpose |
|---|---|---|---|
| OOD Detection | Mahalanobis distance | 97th percentile | Rejects non-retinal or corrupted scans before classification |
| Uncertainty | MC Dropout (20 passes) | σ > 0.15 | Flags low-confidence predictions for specialist review |
| Calibration | Temperature scaling | T ≈ 1.05 | Corrects the systematic overconfidence in raw softmax outputs |
| Explainability | Grad-CAM + SHAP | — | Spatial attribution on the scan + feature importance from XGBoost |

In the NORMAL/DRUSEN case above, the MC Dropout uncertainty exceeded the 0.15 threshold, and the Mahalanobis score flagged the scan as borderline OOD. The case would have been routed to a specialist rather than silently misclassified.

---

## Edge Deployment — Model Compression

The 2.07 GB Keras master model is not deployable in most clinical environments. The edge node solves this.

**What was done:** The mixed-precision Keras model was converted to a clean FP32 ONNX graph via tf2onnx (opset 17), with all FP16 internal tensors cast to FP32 for runtime compatibility. This removes the TensorFlow/Keras dependency entirely for inference.

### Latency Benchmark (Intel CPU, Batch Size 1, 50 runs)

| Scan | Prediction | Confidence | Latency |
|---|---|---|---|
| DRUSEN | DRUSEN ✅ | 92.72% | 61.96 ms |
| DME | DME ✅ | 81.20% | 62.60 ms |
| CNV | CNV ✅ | 91.46% | 64.31 ms |
| NORMAL | DRUSEN ⚠️ | 66.85% | 62.82 ms |
| **Global Average** | — | — | **~62.9 ms** |

The NORMAL misclassification at 66.85% confidence is the reason the uncertainty and OOD mechanisms exist — this scan would be flagged before reaching a clinical report.

### Model Size Comparison

| Format | Size | Reduction |
|---|---|---|
| Keras `.keras` (master) | 2,070 MB | baseline |
| **ONNX FP32 (edge node)** | **237 MB** | **~88% smaller** |

### Run Edge Inference

```powershell
cd edge_inference
python run_inference.py scan_name.jpeg
```

---

## Repository Structure

```
├── deployment_cloud/           # Streamlit dashboard (HuggingFace demo)
│   ├── app.py
│   └── demo_results.json
├── edge_inference/             # Edge inference node
│   ├── run_inference.py        # 62.9ms latency benchmark script
│   ├── convert_to_edge.py      # Keras → ONNX FP32 converter
│   └── human_eye_fp32.onnx     # 237 MB optimised edge model
├── Human-Eye.ipynb             # Full training pipeline — Phases 1–6
├── reproduce/
│   ├── environment.yml         # Conda environment with pinned versions
│   ├── configs/
│   │   └── model_config.yaml   # All hyperparameters
│   └── data_splits/            # Exact train/val/test indices
└── assets/                     # Explainability and statistical plots
```

---

## Setup — Local GPU Inference (RTX 4060)

### Step 1 — Create environment
```bash
conda env create -f reproduce/environment.yml
conda activate oct_retinal
```

### Step 2 — Download model weights
```python
from huggingface_hub import hf_hub_download
import os

os.makedirs('models', exist_ok=True)
for f in ['Final_CNN_Transformer.keras', 'Final_XGBoost_Hybrid.json',
          'ood_train_mean.npy', 'ood_cov_inv.npy',
          'ood_threshold.npy', 'temperature.npy']:
    hf_hub_download(
        repo_id='animeshakr/oct-retinal-weights',
        filename=f, local_dir='models/')
```

### Step 3 — Run dashboard
```bash
DEMO_MODE=false streamlit run deployment_cloud/app.py
```

---

## Dataset

Kermany et al. (Cell 2018) — 84,495 OCT B-scans, 4 classes.

| Class | Train | Test | Clinical Significance |
|---|---|---|---|
| CNV | 37,206 | 3,960 | Wet AMD — urgent anti-VEGF treatment |
| DME | 11,349 | 1,101 | Diabetic macular oedema |
| DRUSEN | 8,617 | 1,086 | Early AMD biomarker — 4.3× class imbalance vs. CNV |
| NORMAL | 26,315 | 1,786 | Healthy retina |

---

## Training Pipeline (Human-Eye.ipynb)

| Phase | Description |
|---|---|
| Phase 1 | Data pipeline — Kermany OCT, augmentation, class weights |
| Phase 2 | EfficientNetV2L + Transformer architecture |
| Phase 3 | Optuna HPO (10 trials, TPE sampler) + Phase A/B training + XGBoost |
| Phase 4 | OOD detection + MC Dropout + temperature calibration |
| Phase 5 | Grad-CAM, SHAP, UMAP, attention maps, ablation, McNemar test |
| Phase 6 | 5-seed statistical validation |

---

## Links

- 🤗 Live demo: https://huggingface.co/spaces/animeshakr/oct-retinal-ai
- 🤗 Model weights: https://huggingface.co/animeshakr/oct-retinal-weights
- 📦 Zenodo archive: https://doi.org/10.5281/zenodo.19224303
- 👤 ORCID: https://orcid.org/0009-0003-0608-7004

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

# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**  
MSc Advanced Computer Science — Newcastle University (2025–26)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19224303.svg)](https://doi.org/10.5281/zenodo.19224303)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![Model Weights](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)
[![Preprint](https://img.shields.io/badge/medRxiv-Preprint-red)](https://www.medrxiv.org/content/10.1101/2026.03.28.349562)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--0608--7004-brightgreen)](https://orcid.org/0009-0003-0608-7004)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

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

> Our model trades ~1% raw accuracy vs. the Kermany baseline to achieve
> clinical-grade probability calibration (ECE=0.0024, 12× improvement)
> and three integrated safety mechanisms absent from all prior work on this dataset.

---

## Ablation Study

| Variant | Accuracy | AUC-ROC |
|---|---|---|
| EfficientNetV2L (frozen) + Dense | 89.12% | 0.9410 |
| + Fine-tuned Block 6+ | 92.45% | 0.9755 |
| + Transformer (4× MHA) | 94.10% | 0.9880 |
| + XGBoost head **(full model)** | **95.43%** | **0.9941** |

Each component contributes independently — no single addition accounts for the full gain.

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

| Feature | Method | Purpose |
|---|---|---|
| OOD Detection | Mahalanobis distance (97th pct) | Rejects non-retinal / corrupt scans |
| Uncertainty | MC Dropout (20 passes) | Flags scans needing specialist review |
| Calibration | Temperature scaling (T≈1.05) | Corrects overconfident probabilities |
| Explainability | Grad-CAM + SHAP | Spatial + feature attribution |

---

## Repository Structure

```
├── app.py                      # Streamlit dashboard (local GPU + HuggingFace demo)
├── generate_demo.py            # Precompute demo results for HuggingFace
├── Human-Eye.ipynb             # Full training pipeline — Phases 1–6
├── requirements.txt            # HuggingFace deployment (lightweight)
├── requirements_local.txt      # Local GPU inference (full stack)
├── reproduce/
│   ├── environment.yml         # Conda environment with pinned versions
│   ├── README.md               # Step-by-step reproduction guide
│   ├── configs/
│   │   └── model_config.yaml   # All hyperparameters as config
│   └── data_splits/
│       ├── train_indices.npy   # Exact split indices used in experiments
│       ├── val_indices.npy
│       └── test_indices.npy
└── assets/
    ├── gradcam_panel.png
    ├── shap_summary.png
    ├── attention_heads_*.png
    ├── umap_2d_features.png
    ├── multiseed_violin.png
    ├── multiseed_aggregate.csv
    └── class_distribution.png
```

---

## Setup — Local GPU Inference (RTX 4060)

### Requirements
- Python **3.11+** (required for TF 2.19 / Keras 3.x)
- NVIDIA GPU with CUDA support

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
DEMO_MODE=false streamlit run app.py
```

---

## Training Pipeline (Human-Eye.ipynb)

| Phase | Description |
|---|---|
| Phase 1 | Data pipeline — Kermany OCT, augmentation, class weights |
| Phase 2 | EfficientNetV2L + Transformer architecture |
| Phase 3 | Optuna HPO (10 trials) + Phase A/B training + XGBoost |
| Phase 4 | OOD detection + MC Dropout + temperature calibration |
| Phase 5 | Grad-CAM, SHAP, UMAP, attention maps, ablation, McNemar test |
| Phase 6 | 5-seed statistical validation |

---

## Dataset

Kermany et al. (Cell 2018) — 84,495 OCT B-scans · 4 classes

| Class | Train | Test | Clinical Significance |
|---|---|---|---|
| CNV | 37,206 | 3,960 | Wet AMD — urgent anti-VEGF treatment |
| DME | 11,349 | 1,101 | Diabetic macular oedema |
| DRUSEN | 8,617 | 1,086 | Early AMD biomarker — 4.3× class imbalance |
| NORMAL | 26,315 | 1,786 | Healthy retina |

---

## Links

- 🤗 Live demo: https://huggingface.co/spaces/animeshakr/oct-retinal-ai
- 🤗 Model weights: https://huggingface.co/animeshakr/oct-retinal-weights
- 📄 Preprint: https://www.medrxiv.org/content/10.1101/2026.03.28.349562
- 📦 Zenodo archive: https://doi.org/10.5281/zenodo.19224303
- 👤 ORCID: https://orcid.org/0009-0003-0608-7004

---

## Citation

```bibtex
@article{kumar2026oct,
  author  = {Kumar, Animesh A.},
  title   = {A Hybrid CNN-Transformer Framework for Retinal OCT
             Classification with Integrated Clinical Safety Mechanisms},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.1101/2026.03.28.349562}
}
```

---

**Author:** Animesh A. Kumar — MSc Advanced Computer Science, Newcastle University (2025–26)  
**License:** MIT

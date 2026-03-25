# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**  
MSc Advanced Computer Science — Newcastle University (2025–26)

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![Model Weights](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--0608--7004-brightgreen)](https://orcid.org/0009-0003-0608-7004)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Results (5-seed statistical validation)

| Metric | Mean ± Std |
|---|---|
| Accuracy | 95.43% ± 0.27% |
| Macro AUC-ROC | 0.9941 ± 0.0006 |
| Macro F1 | 0.9244 ± 0.0047 |
| Drusen F1 (minority class) | 0.8436 ± 0.0096 |
| ECE (calibrated) | 0.0024 ± 0.0005 |
| McNemar p-value | 0.0001 ± 0.0001 (all 5 seeds significant) |

---

## Architecture

```
OCT Scan (224×224×3)
    → EfficientNetV2L backbone (118.5M params, ImageNet pretrained)
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
| OOD Detection | Mahalanobis distance | Rejects non-retinal / corrupt scans |
| Uncertainty | MC Dropout (20 passes) | Flags scans needing specialist review |
| Calibration | Temperature scaling | Corrects overconfident probabilities |
| Explainability | Grad-CAM + SHAP | Shows *why* the model predicts each class |

---

## Repository Structure

```
├── app.py                    # Streamlit dashboard (two-tier: local GPU + HuggingFace demo)
├── generate_demo.py          # Precompute demo results for HuggingFace deployment
├── Human-Eye.ipynb           # Full training pipeline — Phases 1–6
├── requirements.txt          # HuggingFace deployment (lightweight)
├── requirements_local.txt    # Local GPU inference (full stack)
├── setup_repos.py            # Upload weights to HuggingFace + push to GitHub
└── assets/
    ├── gradcam_panel.png
    ├── shap_summary.png
    ├── attention_heads_*.png
    ├── umap_2d_features.png
    ├── umap_3d_features.html
    ├── uncertainty_landscape.html
    ├── multiseed_violin.png
    ├── multiseed_aggregate.csv
    └── class_distribution.png
```

---

## Setup — Local GPU Inference (RTX 4060)

### Requirements
- Python **3.11+** (required for TF 2.19 / Keras 3.x)
- NVIDIA GPU with CUDA support

### Install
```bash
conda create -n oct_dashboard python=3.11 -y
conda activate oct_dashboard
pip install -r requirements_local.txt
```

### Download model weights
```python
from huggingface_hub import hf_hub_download

# Download all required files into models/
files = [
    'Final_CNN_Transformer.keras',
    'Final_XGBoost_Hybrid.json',
    'ood_train_mean.npy',
    'ood_cov_inv.npy',
    'ood_threshold.npy',
    'temperature.npy',
]
for f in files:
    hf_hub_download(
        repo_id='animeshakr/oct-retinal-weights',
        filename=f,
        local_dir='models/'
    )
```

### Run dashboard
```bash
streamlit run app.py
```

---

## Training Pipeline (Human-Eye.ipynb)

| Phase | Description |
|---|---|
| Phase 1 | Data pipeline — Kermany OCT dataset, augmentation, class weights |
| Phase 2 | EfficientNetV2L + Transformer architecture definition |
| Phase 3 | Optuna HPO (10 trials) + Phase A/B training + XGBoost head |
| Phase 4 | OOD detection + MC Dropout uncertainty + temperature calibration |
| Phase 5 | Grad-CAM, SHAP, UMAP, attention maps, ablation, McNemar test |
| Phase 6 | 5-seed statistical validation — mean ± std reporting |

---

## Dataset

Kermany et al. (Cell 2018) — 84,495 OCT B-scans · 4 classes

| Class | Train | Test | Clinical meaning |
|---|---|---|---|
| CNV | 37,206 | 3,960 | Choroidal neovascularisation — wet AMD, urgent treatment needed |
| DME | 11,349 | 1,101 | Diabetic macular edema — anti-VEGF or laser treatment |
| DRUSEN | 8,617 | 1,086 | Early AMD biomarker — lifestyle intervention window |
| NORMAL | 26,315 | 1,786 | Healthy retina |

---

## HuggingFace Deployment

```bash
# Generate precomputed demo results (run once with full model locally)
python generate_demo.py

# Upload demo_results.json + assets/ + app.py + requirements.txt to HuggingFace Space
# Set environment variable in Space settings: DEMO_MODE = true
```

---

## Links

- 🤗 Live demo: https://huggingface.co/spaces/animeshakr/oct-retinal-ai
- 🤗 Model weights: https://huggingface.co/animeshakr/oct-retinal-weights
- 📊 Dataset: [Kermany et al., Cell 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

---

## Citation

```bibtex
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018}
}
```

---

## 👤 Author

**Animesh Kumar**
* MSc Advanced Computer Science, Newcastle University (2025–26)
* ORCID: [0009-0003-0608-7004](https://orcid.org/0009-0003-0608-7004)

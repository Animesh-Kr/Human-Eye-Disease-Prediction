r"""
setup_repos.py
==============
Run this once to:
1. Create HuggingFace model repo and upload weights
2. Push code to GitHub

Usage:
    conda activate oct_dashboard
    cd C:\Users\adim\Desktop\oct_phd_outputs\oct_retinal_dashboard
    python setup_repos.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
HF_USERNAME   = "animeshakr"
HF_MODEL_REPO = "oct-retinal-weights"
GITHUB_REPO   = "https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git"

DASHBOARD_DIR = Path(r"C:\Users\adim\Desktop\oct_phd_outputs\oct_retinal_dashboard")
MODELS_DIR    = DASHBOARD_DIR / "models"
ASSETS_DIR    = DASHBOARD_DIR / "assets"

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: HUGGINGFACE MODEL HUB
# ─────────────────────────────────────────────────────────────────────────────
def upload_to_huggingface():
    print("\n" + "="*60)
    print("PART 1: HuggingFace Model Hub")
    print("="*60)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    api      = HfApi()
    repo_id  = f"{HF_USERNAME}/{HF_MODEL_REPO}"

    # Create model repo
    print(f"\nCreating model repo: {repo_id}")
    try:
        create_repo(
            repo_id   = repo_id,
            repo_type = "model",
            exist_ok  = True,
            private   = False,
        )
        print(f"  Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  Repo may already exist: {e}")

    # Create model card
    model_card = f"""---
language: en
license: mit
tags:
- medical-imaging
- retinal-disease
- oct
- efficientnetv2
- transformer
- xgboost
metrics:
- accuracy
- f1
- auc
---

# OCT Retinal Disease Classification — Model Weights

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**

## Results (5-seed validation)

| Metric | Value |
|---|---|
| Accuracy | 95.43% ± 0.27% |
| Macro AUC | 0.9941 ± 0.0006 |
| Macro F1 | 0.9244 ± 0.0047 |
| Drusen F1 | 0.8436 ± 0.0096 |
| ECE (cal) | 0.0024 ± 0.0005 |

## Files

- `Final_CNN_Transformer.keras` — full model (EfficientNetV2L + 4× MHA)
- `Final_XGBoost_Hybrid.json` — XGBoost head
- `ood_train_mean.npy` / `ood_cov_inv.npy` / `ood_threshold.npy` — OOD detector
- `temperature.npy` — temperature scaling calibration

## Usage

```python
import tensorflow as tf
import keras
import xgboost as xgb
import numpy as np

model = tf.keras.models.load_model(
    'Final_CNN_Transformer.keras',
    custom_objects={{...}}  # see GitHub repo
)
```

## Links
- 🤗 Live Demo: https://huggingface.co/spaces/animeshakr/oct-retinal-ai
- 💻 Code: https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction
- 📊 Dataset: Kermany et al. (Cell 2018)

## Citation
```
Kermany et al. (2018). Identifying Medical Diagnoses and Treatable Diseases
by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
```
"""

    card_path = MODELS_DIR / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    # Files to upload
    files = [
        (MODELS_DIR / "README.md",                    "README.md"),
        (MODELS_DIR / "Final_CNN_Transformer.keras",  "Final_CNN_Transformer.keras"),
        (MODELS_DIR / "Final_XGBoost_Hybrid.json",    "Final_XGBoost_Hybrid.json"),
        (MODELS_DIR / "ood_train_mean.npy",            "ood_train_mean.npy"),
        (MODELS_DIR / "ood_cov_inv.npy",              "ood_cov_inv.npy"),
        (MODELS_DIR / "ood_threshold.npy",            "ood_threshold.npy"),
        (MODELS_DIR / "temperature.npy",              "temperature.npy"),
    ]

    print(f"\nUploading {len(files)} files to {repo_id}...")
    for local_path, repo_path in files:
        if not Path(local_path).exists():
            print(f"  SKIP (not found): {local_path}")
            continue
        size_mb = Path(local_path).stat().st_size / 1e6
        print(f"  Uploading {repo_path} ({size_mb:.1f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj = str(local_path),
                path_in_repo    = repo_path,
                repo_id         = repo_id,
                repo_type       = "model",
            )
            print(f"    Done")
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nModel hub: https://huggingface.co/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: GITHUB
# ─────────────────────────────────────────────────────────────────────────────
def push_to_github():
    print("\n" + "="*60)
    print("PART 2: GitHub")
    print("="*60)

    clone_dir = Path(r"C:\Users\adim\Desktop\oct_github_push")

    # Clone
    print(f"\nCloning {GITHUB_REPO}...")
    if clone_dir.exists():
        import shutil
        shutil.rmtree(clone_dir)
    subprocess.run(["git", "clone", GITHUB_REPO, str(clone_dir)], check=True)
    os.chdir(clone_dir)

    # .gitignore
    gitignore = """models/
demo_results.json
__pycache__/
*.pyc
.streamlit/
sample_scans/
*.egg-info/
.env
*.log
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore)

    # README
    readme = f"""# OCT Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**  
MSc Advanced Computer Science — Newcastle University (2025–26)

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![Model Weights](https://img.shields.io/badge/🤗-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)

## Results (5-seed statistical validation)

| Metric | Mean ± Std |
|---|---|
| Accuracy | 95.43% ± 0.27% |
| Macro AUC-ROC | 0.9941 ± 0.0006 |
| Macro F1 | 0.9244 ± 0.0047 |
| Drusen F1 (minority) | 0.8436 ± 0.0096 |
| ECE (calibrated) | 0.0024 ± 0.0005 |
| McNemar p-value | 0.0001 ± 0.0001 |

## Architecture

```
OCT Scan (224×224×3)
    → EfficientNetV2L (118.5M params, pretrained ImageNet)
    → Patch reshape: 7×7×1280 → 49 tokens
    → Linear projection: 1280 → 256-d
    → Learnable Positional Encoding
    → 4× Multi-Head Attention (16 heads each)
    → GlobalAvgPool1D → 256-d feature vector
    → XGBoost (300 trees, depth=4)
    → Temperature scaling (T≈1.05)
    → CNV / DME / DRUSEN / NORMAL
```

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── generate_demo.py        # Precompute demo results for HuggingFace
├── requirements.txt
├── assets/                 # Explainability outputs (Grad-CAM, SHAP, UMAP)
└── Human-Eye.ipynb         # Full training pipeline (Phases 1-6)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights from HuggingFace
# https://huggingface.co/animeshakr/oct-retinal-weights
# Place in models/ folder

# Run dashboard locally
streamlit run app.py

# Generate HuggingFace demo JSON (run once with full model)
python generate_demo.py
```

## Training Pipeline

| Phase | Description |
|---|---|
| Phase 1 | Data pipeline, augmentation, class weighting |
| Phase 2 | EfficientNetV2L + Transformer architecture |
| Phase 3 | Optuna HPO + Phase A/B training + XGBoost head |
| Phase 4 | OOD detection + MC Dropout + temperature calibration |
| Phase 5 | Grad-CAM, SHAP, UMAP, attention maps, ablation |
| Phase 6 | 5-seed statistical validation |

## Dataset

Kermany et al. (Cell 2018) — 84,495 OCT B-scans · 4 classes (CNV, DME, DRUSEN, NORMAL)

## Links

- 🤗 Live demo: https://huggingface.co/spaces/animeshakr/oct-retinal-ai
- 🤗 Model weights: https://huggingface.co/animeshakr/oct-retinal-weights
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # Copy files from dashboard
    import shutil

    files_to_copy = [
        (DASHBOARD_DIR / "app.py",                        "app.py"),
        (DASHBOARD_DIR / "generate_demo.py",              "generate_demo.py"),
        (DASHBOARD_DIR / "requirements.txt",              "requirements.txt"),
        (DASHBOARD_DIR / "requirements_huggingface.txt",  "requirements_huggingface.txt"),
        (DASHBOARD_DIR / "Dockerfile",                    "Dockerfile"),
    ]
    
    # Copy training notebook if present
    notebook_src = Path(r"C:\Users\adim\Desktop\oct_phd_outputs") / "Human-Eye.ipynb"
    if notebook_src.exists():
        shutil.copy2(notebook_src, "Human-Eye.ipynb")
        print("  Copied: Human-Eye.ipynb (training pipeline)")

    for src, dst in files_to_copy:
        if Path(src).exists():
            shutil.copy2(src, dst)
            print(f"  Copied: {dst}")

    # Copy assets folder
    if ASSETS_DIR.exists():
        assets_dst = clone_dir / "assets"
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(ASSETS_DIR, assets_dst)
        print(f"  Copied: assets/ ({len(list(assets_dst.iterdir()))} files)")

    # Git add + commit + push
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m",
        "feat: OCT retinal AI — EfficientNetV2L + 4x MHA + XGBoost\n\n"
        "Macro AUC: 0.9941 +/- 0.0006 | Accuracy: 95.43% +/- 0.27% | 5-seed validation\n"
        "Live demo: https://huggingface.co/spaces/animeshakr/oct-retinal-ai\n"
        "Weights: https://huggingface.co/animeshakr/oct-retinal-weights"
    ], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)

    print(f"\nGitHub: {GITHUB_REPO.replace('.git', '')}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("OCT Retinal AI — Repository Setup")
    print("This will:")
    print("  1. Create HuggingFace model repo and upload weights (~2GB, 5-10 min)")
    print("  2. Push code to GitHub")
    print()

    choice = input("Run both? (y/n): ").strip().lower()
    if choice != 'y':
        print("Aborted.")
        exit()

    upload_to_huggingface()
    push_to_github()

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    print(f"HuggingFace weights: https://huggingface.co/animeshakr/oct-retinal-weights")
    print(f"HuggingFace demo:    https://huggingface.co/spaces/animeshakr/oct-retinal-ai")
    print(f"GitHub:              https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction")

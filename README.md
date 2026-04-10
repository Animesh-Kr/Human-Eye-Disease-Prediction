# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid** MSc Advanced Computer Science — Newcastle University (2025–26)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19224303.svg)](https://doi.org/10.5281/zenodo.19224303)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
[![Model Weights](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-retinal-weights)
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
| Kermany et al. [cite_start]2018 [cite: 127]| InceptionV3 | 96.6% | — | — | None |
| He et al. [cite_start]2019 [cite: 116, 117]| DenseNet/ResNet | 93.2% | — | — | None |
| Li et al. [cite_start]2021 [cite: 130, 131]| Attention DenseNet | 91.7% | 0.97 | — | None |
| **Ours** | **EfficientNetV2L + 4×MHA + XGBoost** | **95.43%** | **0.9941** | **0.0024** | **OOD + Uncertainty + Calibration** |

> Our model trades ~1% raw accuracy vs. the Kermany baseline to achieve 
> clinical-grade probability calibration (ECE=0.0024, 12× improvement) 
> [cite_start]and three integrated safety mechanisms absent from all prior work on this dataset. [cite: 10, 123]

---

## Evaluation Results (Baseline Comparison)

| Model | Accuracy | Macro AUC-ROC | Macro F1 | ECE (Calibrated) |
|---|---|---|---|---|
| [cite_start]Baseline CNN (EffNetV2L + Dense) [cite: 48, 66] | 89.12% | 0.9410 | 0.8642 | 0.0203 |
| [cite_start]**Hybrid CNN-Transformer-XGBoost** [cite: 62, 66] | **95.43%** | **0.9941** | **0.9244** | **0.0024** |

---

## Ablation Study

| Variant | Accuracy | AUC-ROC |
|---|---|---|
| [cite_start]EfficientNetV2L (frozen) + Dense [cite: 66]| 89.12% | 0.9410 |
| + [cite_start]Fine-tuned Block 6+ [cite: 66]| 92.45% | 0.9755 |
| + [cite_start]Transformer (4× MHA) [cite: 66]| 94.10% | 0.9880 |
| + [cite_start]XGBoost head **(full model)** [cite: 66]| **95.43%** | **0.9941** |

[cite_start]Each component contributes independently — no single addition accounts for the full gain. [cite: 64]

---

## Architecture

```
OCT Scan (224×224×3)
    [cite_start]→ EfficientNetV2L backbone (118.5M params, ImageNet-21k pretrained) [cite: 51]
        Blocks 1–5: frozen  |  [cite_start]Block 6+: fine-tuned [cite: 52]
    [cite_start]→ Patch reshape: 7×7×1280 → 49 tokens [cite: 52]
    [cite_start]→ Linear projection: 1280 → 256-d [cite: 52]
    [cite_start]→ Learnable Positional Encoding [cite: 52]
    [cite_start]→ 4× Multi-Head Attention (16 heads, key_dim=16) [cite: 53]
    [cite_start]→ GlobalAvgPool1D → 256-d feature vector [cite: 54]
    [cite_start]→ XGBoost hybrid head (300 trees, max_depth=4) [cite: 57]
    [cite_start]→ Temperature scaling (T≈1.05) [cite: 59]
    → CNV / DME / DRUSEN / NORMAL
```

---

## Clinical Safety Features

| Feature | Method | Purpose |
|---|---|---|
| OOD Detection | [cite_start]Mahalanobis distance (97th pct) [cite: 60, 61]| [cite_start]Rejects non-retinal / corrupt scans [cite: 38]|
| Uncertainty | [cite_start]MC Dropout (20 passes) [cite: 39]| [cite_start]Flags scans needing specialist review [cite: 40]|
| Calibration | [cite_start]Temperature scaling (T≈1.05) [cite: 59]| [cite_start]Corrects overconfident probabilities [cite: 30]|
| Explainability | [cite_start]Grad-CAM + SHAP [cite: 23]| [cite_start]Spatial + feature attribution [cite: 41, 42]|

---

## Repository Structure

```
├── deployment_cloud/           # Streamlit dashboard (HuggingFace demo)
│   ├── app.py
│   └── demo_results.json
├── edge_inference/             # High-speed Edge Node
│   ├── run_inference.py        # 62ms latency benchmark script
│   ├── convert_to_edge.py      # Keras-to-ONNX FP32 converter
│   └── human_eye_fp32.onnx     # Optimized 237MB Edge model
├── Human-Eye.ipynb             # Full training pipeline — Phases 1–6
├── reproduce/
│   ├── environment.yml         # Conda environment with pinned versions
│   └── data_splits/            # Exact split indices used in experiments
└── assets/                     # Explainability & statistical plots
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

---

# Model Compression & Edge Deployment

## ONNX Export + FP32 Optimization

The master Keras framework was distilled into a high-performance **FP32 ONNX graph**. While INT8 quantization was evaluated, the FP32 Edge Node was selected to preserve graph stability and parity with clinical safety mechanisms.

### Inference Latency Benchmark (Edge Node)

> Hardware: Intel Core CPU (Local Node) · Batch size: 1 · 50 runs

| Backend | Mean (ms) | P50 (ms) | Speedup | Notes |
|---|---|---|---|---|
| Keras Master (CPU) | ~400.0 | ~395.0 | baseline | Full TF/Keras stack |
| **ONNX Runtime (CPU FP32)** | **62.4** | **61.9** | **~6.5×** | **Optimized Edge Node** |

### Model Size Comparison

| Format | Size | Reduction |
|---|---|---|
| Keras `.keras` (original) | 2,070 MB | baseline |
| **ONNX FP32 (Optimized)** | **237 MB** | **~88% smaller** |

### Run Edge Inference

```powershell
# Navigate to edge folder
cd edge_inference

# Run benchmark on any scan
python run_inference.py scan_name.jpeg
```

---

## Dataset

Kermany et al. (Cell 2018) [cite_start][cite: 36, 127] — 84,495 OCT B-scans · 4 classes

| Class | Train | Test | [cite_start]Clinical Significance [cite: 45]|
|---|---|---|---|
| CNV | 37,206 | 3,960 | Wet AMD — urgent anti-VEGF treatment |
| DME | 11,349 | 1,101 | Diabetic macular oedema |
| DRUSEN | 8,617 | 1,086 | Early AMD biomarker — 4.3× class imbalance |
| NORMAL | 26,315 | 1,786 | Healthy retina |

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
```

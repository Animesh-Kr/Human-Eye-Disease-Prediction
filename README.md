---
title: OCT Retinal AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
license: mit
doi: 10.5281/zenodo.19224304
short_description: CNN-Transformer hybrid for retinal disease classification
---

# 👁️ OCT Retinal AI — Retinal Disease Classification

**EfficientNetV2L + 4× Multi-Head Attention + XGBoost Hybrid**

A state-of-the-art CNN-Transformer hybrid model for automated OCT retinal disease classification, built as part of an MSc dissertation at Newcastle University (2025–26). This project bridges the gap between massive cloud-based training and high-performance edge deployment.

## Model Performance (5-seed validation)

| Metric | Value |
|---|---|
| Accuracy | 95.43% ± 0.27% |
| Macro AUC-ROC | 0.9941 ± 0.0006 |
| Macro F1 | 0.9244 ± 0.0047 |
| Drusen F1 (minority class) | 0.8436 ± 0.0096 |
| ECE (calibrated) | 0.0024 ± 0.0005 |
| McNemar p-value | < 0.0001 (all seeds) |

## Architecture

- [cite_start]**Backbone:** EfficientNetV2L (118.5M params, ImageNet pretrained) [cite: 51]
- [cite_start]**Transformer:** 4× Multi-Head Attention blocks (16 heads, proj_dim=256) 
- [cite_start]**Positional Encoding:** Learnable (not sinusoidal) [cite: 52]
- [cite_start]**Head:** XGBoost (300 trees, max_depth=4) 
- [cite_start]**Safety:** Mahalanobis OOD detection (97th percentile threshold) [cite: 38, 60]
- [cite_start]**Calibration:** Temperature scaling (T≈1.05) [cite: 59]
- [cite_start]**Uncertainty:** MC Dropout (20 stochastic forward passes) 

## ⚡ Edge Node Optimization (New)

To ensure clinical utility in resource-constrained settings, the master model was distilled into a high-performance Edge Node using ONNX. This optimization removes the heavy dependency on TensorFlow/Keras for inference while maintaining the hybrid CNN-Transformer graph integrity.

- **Model Compression:** 2.07 GB (Keras Master) → **237 MB (FP32 ONNX Node)**.
- **Footprint Reduction:** **~88% storage reduction** for edge deployment.
- **Inference Latency:** **~62.4 ms per scan** (Benchmark: Standard Intel CPU, Batch Size 1).
- **Deployment:** Decoupled inference pipeline utilizing a standardized ImageNet-normalization and RGB-conversion layer to ensure parity with the cloud-trained Master node.

## Dashboard Features

- [cite_start]**Clinical Workspace** — Upload scan → OOD check → prediction → calibrated confidence → Grad-CAM → MC Dropout uncertainty [cite: 23]
- **3D Architecture** — Interactive Plotly 3D network topology
- [cite_start]**Explainability** — Grad-CAM panel · SHAP feature importance · Transformer attention head maps [cite: 41, 42]
- **Feature Space** — Interactive UMAP 3D clusters · Uncertainty landscape
- [cite_start]**Phase 6 Validation** — Multi-seed mean ± std results · Violin plots [cite: 49, 112]

## Dataset

Kermany et al. (Cell 2018) [cite_start]— 84,495 OCT B-scans · 4 classes (CNV, DME, DRUSEN, NORMAL) [cite: 36]

## Demo Mode

This Space runs in demo mode using 20 precomputed sample scans (5 per class). 
Select a scan from the sidebar to see the full inference pipeline output instantly. To test the **Live Edge Inference Node**, please refer to the `edge_inference/` directory in the GitHub repository.

## Author

Animesh Kumar — MSc Advanced Computer Science, Newcastle University (2025–26)

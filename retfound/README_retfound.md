# RETFound Fine-Tuning — OCT Retinal Disease Classification

**Branch:** `retfound-finetune`  
**Base model:** RETFound (ViT-L/16, MAE pretraining on 1.6M retinal images)  
**Task:** 4-class OCT classification — CNV / DME / DRUSEN / NORMAL  
**Dataset:** Kermany et al. (Cell 2018) — 84,495 OCT B-scans  
**Paper:** Zhou et al., *Nature* 2023. https://doi.org/10.1038/s41586-023-06555-x

---

## What Is RETFound

RETFound is a foundation model for retinal images developed by Moorfields Eye Hospital (UCL), pretrained using **Masked Autoencoder (MAE)** on 1.6 million unlabelled retinal scans. This experiment quantifies how much of our performance comes from architectural design vs. pretraining domain.

---

## Experimental Results

### Linear Probe (Frozen Encoder)
> Seeds: 42, 123, 2024 | Epochs: 20 | Head only trained

| Metric | RETFound Linear Probe | Main Branch |
|---|---|---|
| Accuracy | 0.7572 ± 0.0067 | 0.9543 ± 0.0027 |
| Macro AUC-ROC | 0.9103 ± 0.0002 | 0.9941 ± 0.0006 |
| Macro F1 | 0.6590 ± 0.0052 | 0.9244 ± 0.0047 |
| Drusen F1 | 0.3829 ± 0.0009 | 0.8436 ± 0.0096 |

### Full Fine-Tuning (All 24 Transformer Blocks)
> Seeds: 42, 123, 999 | Epochs: 50 | Layer-wise LR decay 0.65

| Metric | RETFound Full FT | Main Branch |
|---|---|---|
| Accuracy | 0.9514 ± 0.0137 | **0.9543 ± 0.0027** |
| Macro AUC-ROC | 0.9922 ± 0.0041 | **0.9941 ± 0.0006** |
| Macro F1 | 0.9189 ± 0.0214 | **0.9244 ± 0.0047** |
| Drusen F1 | 0.8192 ± 0.0420 | **0.8436 ± 0.0096** |

### Per-Seed — Full Fine-Tuning

| Seed | Accuracy | AUC-ROC | Drusen F1 |
|---|---|---|---|
| 42 | 0.9633 | 0.9952 | 0.8559 |
| 123 | 0.9587 | 0.9951 | 0.8414 |
| 999 | 0.9322 | 0.9864 | 0.7604 |
| **Mean ± Std** | **0.9514 ± 0.0137** | **0.9922 ± 0.0041** | **0.8192 ± 0.0420** |

---

## Key Findings

**1. Domain gap matters for frozen encoders.** Linear probing yields only 75.72% — the modality gap between fundus photographs (RETFound pretraining) and OCT B-scans is large enough that a frozen encoder cannot generalise. Full fine-tuning bridges this gap.

**2. Full fine-tuning is competitive but not conclusively superior.** Seeds 42 and 123 both exceeded the main branch (96.33%, 95.87% vs 95.43%). Seed 999 underperformed (93.22%) due to configuration differences. Mean result (95.14%) is statistically comparable to our main branch (95.43%).

**3. Higher variance is the real cost.** Main branch variance ±0.27% vs RETFound ±1.37% — our custom architecture is 5× more stable across seeds.

**4. DRUSEN class favours our architectural design.** The XGBoost hybrid head and inverse-frequency class weighting specifically address the minority DRUSEN class in a way that ViT fine-tuning alone does not replicate (0.8436 vs 0.8192).

---

## Full Comparison Table

| Method | Accuracy | AUC-ROC | Macro F1 | Drusen F1 |
|---|---|---|---|---|
| Baseline CNN (EffNetV2L + Dense) | 89.12% | 0.9410 | 0.8642 | — |
| **Ours (EffNetV2L + 4×MHA + XGBoost)** | **95.43% ± 0.27%** | **0.9941 ± 0.0006** | **0.9244 ± 0.0047** | **0.8436 ± 0.0096** |
| RETFound linear probe | 75.72% ± 0.67% | 0.9103 ± 0.0002 | 0.6590 ± 0.0052 | 0.3829 ± 0.0009 |
| RETFound full fine-tuning | 95.14% ± 1.37% | 0.9922 ± 0.0041 | 0.9189 ± 0.0214 | 0.8192 ± 0.0420 |

---

## Training Configuration

| Parameter | Linear Probe | Full Fine-Tuning |
|---|---|---|
| Trainable params | ~4K | 303.3M |
| Epochs | 20 | 50 |
| Learning rate | 1e-3 | 5e-5 |
| LR decay (per block) | N/A | 0.65 |
| Batch size | 128 (H100) | 128 (H100) |
| Gradient accumulation | — | 2 steps (eff. 256) |
| AMP precision | bfloat16 | bfloat16 |
| Seeds | 42, 123, 2024 | 42, 123, 999 |

---

## Files in This Branch

```
retfound/
├── README_retfound.md
├── retfound_oct_finetune.py              ← Full Colab training notebook
├── model_config_retfound.yaml            ← Hyperparameters
├── results_full_finetune_3seed.json      ← 3-seed test results
├── results_linear_probe_3seed.json       ← Linear probe results
├── results_full_finetune_4panel.png      ← Per-seed comparison plots
├── results_linear_probe_4panel.png       ← Linear probe plots
├── tsne_full_finetune.png                ← Latent space (full FT)
├── tsne_linear_probe.png                 ← Latent space (linear probe)
└── calibration_full_finetune.png         ← Reliability diagram
```

---

## Citation

```bibtex
@article{zhou2023foundation,
  title   = {A foundation model for generalizable disease detection
             from retinal images},
  author  = {Zhou, Yukun and Chia, Mark A. and Wagner, Siegfried K.
             and Ayhan, Murat S. and others},
  journal = {Nature},
  volume  = {622},
  pages   = {156--163},
  year    = {2023},
  doi     = {10.1038/s41586-023-06555-x}
}
```

**Author:** Animesh A. Kumar — MSc Advanced Computer Science, Newcastle University (2025–26)

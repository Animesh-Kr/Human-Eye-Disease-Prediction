# 👁️ OCT Retinal AI: Hybrid CNN-Transformer Platform

[](https://www.google.com/search?q=https://huggingface.co/animeshakr/oct-retinal-weights)
[](https://www.tensorflow.org/)
[](https://opensource.org/licenses/MIT)

A high-performance **Hybrid Deep Learning** platform for automated retinal disease diagnosis using Optical Coherence Tomography (OCT) scans. This project implements a state-of-the-art **EfficientNetV2L + Multi-Head Attention + XGBoost** pipeline.

## 🧠 Model Architecture

This system utilizes a three-stage hybrid classification pipeline designed for maximum clinical reliability:

1.  **Spatial Backbone**: **EfficientNetV2-Large** extracts 1280 high-dimensional feature vectors. Blocks 1–5 are frozen to preserve general medical features, while Block 6+ is fine-tuned for retinal pathology.
2.  **Global Context (Transformer)**: A **4-Block Multi-Head Attention** module with **Learnable Positional Encoding** captures long-range dependencies across retinal layers.
3.  **Decision Head**: Features are fed into an **XGBoost** classifier (300 trees) to refine decision boundaries and handle class imbalances (e.g., Drusen).

## 🚀 Key Features

  * **Explainable AI (XAI)**: Integrated **Grad-CAM** and **SHAP** for spatial and feature-based transparency.
  * **Uncertainty Quantization**: Uses **MC Dropout** to provide a confidence score for every diagnosis.
  * **OOD Safety**: **Mahalanobis Distance** check to detect and flag non-retinal or corrupt scans.
  * **RTX Optimized**: Custom memory management optimized for **NVIDIA RTX 4060** hardware.

## 📦 Model Weights (Hugging Face)

Due to the large size of the high-fidelity model (\~2.07 GB), weights are hosted on the Hugging Face Model Hub.

  * **Repository**: [animeshakr/oct-retinal-weights](https://www.google.com/search?q=https://huggingface.co/animeshakr/oct-retinal-weights)
  * **Contents**: `.keras` full model, `.weights.h5` legacy weights, XGBoost JSON, and OOD calibration `.npy` files.

## 📊 Performance

| Metric | Accuracy | Macro AUC | Macro F1 | ECE (Cal) |
| :--- | :--- | :--- | :--- | :--- |
| **Result** | **95.9%** | **0.9947** | **0.9316** | **0.0017** |

## 💻 Installation & Usage

### 1\. Set up the Environment

```bash
conda create -n GPU_RTX python=3.10
conda activate GPU_RTX
pip install tensorflow==2.15.0 xboost opencv-python pandas streamlit albumentations
```

### 2\. Download Weights

You can use the provided Python script to download weights directly from Hugging Face:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='animeshakr/oct-retinal-weights', filename='Final_CNN_Transformer_weights.weights.h5', local_dir='models/')"
```

### 3\. Launch the Dashboard

```bash
streamlit run app.py
```

## 📂 Repository Structure

  * `app.py`: High-fidelity Streamlit dashboard.
  * `app_utils.py`: Hybrid architecture and Grad-CAM logic.
  * `assets/`: Multi-seed validation charts and UMAP visualizations.
  * `models/`: (Ignored by Git) Directory for weights and calibration files.

## 🎓 Academic Context

This research was initiated during a **B.Tech** at **AKTU, Lucknow** and further refined during an **MSc in Advanced Computer Science** at **Newcastle University**.

-----

**Developer**: [Animesh Kumar](https://www.google.com/search?q=https://www.linkedin.com/in/animeshakumar/)

http://googleusercontent.com/interactive_content_block/0

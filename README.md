# Human Eye Disease Prediction: Retinal OCT Analysis Platform 👁️

## 📌 Project Overview

This project implements a high-performance **Hybrid Deep Learning** analysis platform designed to automate the diagnosis of retinal diseases using Optical Coherence Tomography (OCT) scans.

By combining the spatial feature extraction capabilities of **EfficientNetV2** with the robust gradient-boosted decision logic of **XGBoost**, this platform provides a reliable tool for distinguishing between **Normal** retinas and critical pathologies like **CNV**, **DME**, and **Drusen**.

## 🚀 Key Features

* **Hybrid Architecture**: Utilizes a dual-model approach (CNN + XGBoost) for superior diagnostic accuracy.
* **Explainable AI (XAI)**: Integrated **Grad-CAM** visualization provides "heatmaps" to show medical professionals exactly where the AI is focusing its attention.
* **High-Volume Processing**: Optimized to handle large-scale clinical datasets (>84,000 images).
* **Clinical Decision Support**: Designed with medical-grade insights and recommendations for each identified pathology.
* **RTX Optimized**: Custom configuration for hardware acceleration on NVIDIA RTX 40-series GPUs.

## 🧠 Model Architecture

The system utilizes a multi-stage classification pipeline:

1. **Spatial Feature Extraction**: A pre-trained **EfficientNetV2-B0** (fine-tuned) extracts 1280 unique deep-feature vectors from each scan.
2. **Gradient Boosted Classification**: An **XGBoost** classifier processes these high-dimensional vectors to make the final diagnostic decision.
3. **Explainability Layer**: **Grad-CAM** generates activation maps to validate the biological relevance of the AI's findings.

## 🩺 Supported Pathologies

| Disease | Description | Visual Characteristic |
| --- | --- | --- |
| **CNV** | Choroidal Neovascularization | Neovascular membrane with subretinal fluid. |
| **DME** | Diabetic Macular Edema | Retinal thickening with intraretinal fluid. |
| **DRUSEN** | Early AMD | Presence of multiple yellow drusen deposits. |
| **NORMAL** | Healthy Retina | Preserved foveal contour without fluid/edema. |

## 📂 Dataset Information

* **Total Images**: 84,495 High-Resolution scans.
* **Verification**: Tiered expert verification by senior retinal specialists.
* **Source**: [Kaggle - Labeled Optical Coherence Tomography (OCT)](https://www.google.com/search?q=https://www.kaggle.com/paultimothymooney/kermany2018)

## 🛠️ Technical Stack

* **Languages**: Python 3.10
* **Deep Learning**: TensorFlow 2.10, Keras
* **Machine Learning**: XGBoost, Scikit-Learn
* **Computer Vision**: OpenCV (CV2)
* **Deployment**: Streamlit
* **Data Handling**: NumPy, Pandas

## 💻 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git
cd Human-Eye-Disease-Prediction

```

### 2. Set up the Environment

```bash
conda create -n GPU_RTX python=3.10
conda activate GPU_RTX
pip install -r requirements.txt

```

### 3. Run the Training Pipeline

To retrain the hybrid model on your hardware:

```bash
python train_hybrid.py

```

### 4. Launch the Diagnostic Dashboard

To run the interactive Streamlit web application:

```bash
streamlit run app.py

```

## 📊 Results

* **Training Accuracy**: ~97.3%
* **Validation Accuracy**: ~95.0%
* **Hardware Benchmark**: Optimized for NVIDIA RTX 4060 using Stable Mode (Float32).

## 🤝 Contribution

This project was developed as part of MSc research at **Newcastle University**. Contributions for improving model interpretability or expanding the dataset are welcome.

## 📧 Contact

**Developer**: Animesh Kumar

**Education**: MSc Advanced Computer Science, Newcastle University

**Email**: [kranimesh2004@gmail.com](mailto:kranimesh2004@gmail.com)

**LinkedIn**: [Animesh Kumar](https://www.google.com/search?q=https://www.linkedin.com/in/animesh-kumar/)

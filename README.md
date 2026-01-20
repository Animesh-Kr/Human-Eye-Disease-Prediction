# Human Eye Disease Prediction: Retinal OCT Analysis Platform 👁️

## 📌 Project Overview

This project implements a high-performance **Hybrid Deep Learning** analysis platform designed to automate the diagnosis of retinal diseases using Optical Coherence Tomography (OCT) scans.

By combining the spatial feature extraction capabilities of **EfficientNetV2** with the robust gradient-boosted decision logic of **XGBoost**, this platform provides a reliable tool for distinguishing between **Normal** retinas and critical pathologies like **CNV**, **DME**, and **Drusen**.

## 🚀 Key Features

* **Hybrid Architecture**: Utilizes a dual-model approach (CNN + XGBoost) for superior diagnostic accuracy.
* **Medical Preprocessing**: Integrated **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to sharpen pathological markers like subretinal fluid.
* **Dynamic Hardware Detection**: Automatically detects and utilizes available GPU acceleration (e.g., NVIDIA RTX 4060) or falls back to CPU.
* **Clinical Decision Support**: Designed with medical-grade insights and recommendations for each identified pathology.
* **Memory Optimized**: Custom VRAM management to prevent "Paging File" errors on consumer-grade hardware.

## 🧠 Model Architecture

The system utilizes a multi-stage classification pipeline:

1. **Spatial Feature Extraction**: A pre-trained **EfficientNetV2-B0** extracts 1280 unique deep-feature vectors from each scan.
2. **Gradient Boosted Classification**: An **XGBoost** classifier processes these high-dimensional vectors to make the final diagnostic decision.
3. **Preprocessing Layer**: **CLAHE** enhancement is applied to input images to improve contrast before feature extraction.

## 🩺 Supported Pathologies

| Disease | Description | Visual Characteristic |
| --- | --- | --- |
| **CNV** | Choroidal Neovascularization | Neovascular membrane with subretinal fluid. |
| **DME** | Diabetic Macular Edema | Retinal thickening with intraretinal fluid. |
| **DRUSEN** | Early AMD | Presence of multiple yellow drusen deposits. |
| **NORMAL** | Healthy Retina | Preserved foveal contour without fluid/edema. |

## 📂 Dataset Information

* **Total Images**: 84,495 High-Resolution scans.
* **Structure**: 76,515 training images and 10,933 test images.
* **Source**: [Kaggle - Labeled Optical Coherence Tomography (OCT)](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct).

## 🛠️ Technical Stack

* **Languages**: Python 3.10
* **Deep Learning**: TensorFlow 2.10, Keras
* **Machine Learning**: XGBoost 2.x, Scikit-Learn
* **Computer Vision**: OpenCV (CV2)
* **Deployment**: Streamlit 1.24.0

## 💻 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git
cd Human-Eye-Disease-Prediction

```

### 2. Set up the Environment

Due to specific version requirements for GPU acceleration, use the following pinned versions:

```bash
conda create -n GPU_RTX python=3.10
conda activate GPU_RTX
pip install tensorflow==2.10.0 streamlit==1.24.0 protobuf==3.20.3 xgboost opencv-python pandas

```

### 3. Run the Training Pipeline

To retrain the hybrid model on your local hardware (requires dataset in `.cache` or project folder):

```bash
python train_hybrid.py

```

### 4. Launch the Diagnostic Dashboard

To run the interactive Streamlit web application:

```bash
python -m streamlit run app.py

```

## 📊 Results (Final Test Set)

* **Overall Accuracy**: 93%
* **CNV Precision**: 94%
* **Normal Precision**: 95%
* **Hardware Benchmark**: Tested on **NVIDIA GeForce RTX 4060 Laptop GPU**.

## 🤝 Contribution

This project was developed as part of MSc research at **Newcastle University**.

## 📧 Contact

**Developer**: Animesh Kumar

**Education**: MSc Advanced Computer Science, Newcastle University

**LinkedIn**: [Animesh Kumar](https://www.google.com/search?q=https://www.linkedin.com/in/animeshakumar/)

---

**Would you like me to help you create a `requirements.txt` file that matches these pinned versions perfectly?**

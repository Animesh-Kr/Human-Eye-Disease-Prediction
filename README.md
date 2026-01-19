# Human Eye Disease Prediction: Retinal OCT Analysis Platform 👁️

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Maintained-green)]()

## 📌 Project Overview
Optical Coherence Tomography (OCT) is a powerful imaging technique that provides high-resolution cross-sectional images of the retina. This project implements a **Deep Learning based automated analysis platform** designed to streamline the diagnosis of retinal diseases.

By leveraging **Convolutional Neural Networks (CNNs)**, this tool classifies OCT scans into four distinct categories, aiding ophthalmologists in the early detection of conditions that can lead to vision loss, such as Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Age-Related Macular Degeneration (AMD).

## 🚀 Key Features
* **Automated Image Analysis:** Utilizes state-of-the-art CNN architectures to classify OCT images with high accuracy.
* **Multi-Class Classification:** Distinguishes between **Normal**, **CNV**, **DME**, and **Drusen**.
* **High-Volume Processing:** Trained on a dataset of over **84,000** high-resolution images.
* **Medical Decision Support:** Designed to reduce the manual interpretation burden on medical professionals.

## 🩺 Understanding Retinal Diseases
The model is trained to identify the following specific pathologies:

| Disease | Description | Visual Characteristic |
| :--- | :--- | :--- |
| **CNV (Choroidal Neovascularization)** | Neovascular membrane with subretinal fluid. | Abnormal blood vessel growth. |
| **DME (Diabetic Macular Edema)** | Retinal thickening with intraretinal fluid. | Fluid accumulation due to diabetes. |
| **Drusen (Early AMD)** | Presence of multiple drusen deposits. | Yellow deposits under the retina. |
| **Normal** | Preserved foveal contour. | Absence of fluid or edema. |

## 📂 Dataset Information
The model was trained and validated on a large-scale dataset sourced from varied medical centers to ensure patient diversity.

* **Total Images:** 84,495 High-Resolution OCT Scans (JPEG)
* **Structure:** Organized into Train, Test, and Validation sets.
* **Verification:** Images underwent tiered expert verification to ensure ground-truth accuracy.
* **Source:** [Kaggle - Labeled Optical Coherence Tomography (OCT)](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct)

## 🛠️ Technical Stack
* **Core Framework:** Python, TensorFlow/Keras
* **Model Architecture:** Convolutional Neural Networks (CNN) / Transfer Learning (e.g., VGG16/ResNet)
* **Preprocessing:** Image Resizing, Normalization, CLAHE (Contrast Limited Adaptive Histogram Equalization)
* **Data Augmentation:** Rotation, Zoom, Horizontal Flip to prevent overfitting.

## 💻 Installation & Usage

1. **Clone the repository:**
Install dependencies:

Bash

pip install -r requirements.txt
Run the prediction script:

Bash

python predict.py --image_path path/to/your/oct_scan.jpg
📊 Results
Training Accuracy: ~96% (Example metric - update with your real number)

Validation Accuracy: ~94% (Example metric - update with your real number)

Loss: Optimized using Categorical Crossentropy.

🤝 Contribution
Contributions are welcome! Please feel free to submit a Pull Request.

📧 Contact
For questions or collaboration regarding this research:

Developer: Animesh Kumar

Email: kranimesh2004@gmail.com

LinkedIn: Animesh Kumar


### **Important Next Steps for This Repo:**
1.  **Screenshots:** Since the text mentions "Explore Results," you **must** add a folder named `screenshots` to your repo and upload 1-2 images of the code running or a graph of the training accuracy.
2.  **Requirements File:** Make sure you create a file named `requirements.txt` in the repo with the libraries you used (e.g., `tensorflow`, `numpy`, `pandas`, `matplotlib`).
3.  **Typos Fixed:** I corrected "Precdiction" to "Prediction" in the title. Ensure you
   ```bash
   git clone [https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git)
   cd Human-Eye-Disease-Prediction

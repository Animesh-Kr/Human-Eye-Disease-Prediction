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
   ```bash
   git clone [https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction.git)
   cd Human-Eye-Disease-Prediction
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the prediction script:**
   ```bash
   python predict.py --image_path path/to/your/oct_scan.jpg
   
📊 Results
Training Accuracy: ~96% (Update this with your actual final accuracy)

Validation Accuracy: ~94% (Update this with your actual final accuracy)

Loss: Optimized using Categorical Crossentropy.

🤝 Contribution
Contributions are welcome! Please feel free to submit a Pull Request.

📧 Contact
For questions or collaboration regarding this research:

Developer: Animesh Kumar

Email: kranimesh2004@gmail.com

LinkedIn: Animesh Kumar


### **Quick Checklist Before You Commit:**
1.  **Repo Name:** Make sure your repository is actually named `Human-Eye-Disease-Prediction` (check for typos like "Precdiction" in the URL). If the URL is different, update the `git clone` line in the code above.
2.  **Screenshots:** Create a folder named `screenshots` in your repo and upload an image of your model's confusion matrix or a test prediction. (You can then add `![Result Screenshot](screenshots/result.png)` to the Results section later).
3.  **Requirements:** Ensure you have a `requirements.txt` file in the main folder.

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import xgboost as xgb
import pickle
import tempfile
import cv2
import os
from recommendation import cnv, dme, drusen, normal

#  1. SETTINGS & ASSETS LOAD
@st.cache_resource
def load_hybrid_assets():
    """Load models once and cache them in memory."""
    # Load CNN for feature extraction
    full_model = tf.keras.models.load_model("Final_OCT_EfficientNet.keras")
    feature_extractor = tf.keras.models.Model(
        inputs=full_model.input, 
        outputs=full_model.get_layer("feature_extraction_layer").output
    )
    
    # Load XGBoost Brain
    xgb_brain = xgb.XGBClassifier()
    xgb_brain.load_model("Final_XGBoost_Classifier.json")
    
    # Load Class Labels
    with open('class_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
        
    return feature_extractor, xgb_brain, labels

#  2. PREDICTION LOGIC
def hybrid_prediction(image_path):
    feature_extractor, xgb_brain, class_names = load_hybrid_assets()
    
    # 1. Load Image and convert to Grayscale for CLAHE
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # 2. Apply Medical-Grade Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    # 3. Convert back to RGB for EfficientNetV2 and Resize
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 4. Hybrid Prediction Pipeline
    x = tf.cast(img_resized, tf.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    deep_features = feature_extractor.predict(x, verbose=0)
    preds_proba = xgb_brain.predict_proba(deep_features)
    
    result_index = np.argmax(preds_proba)
    confidence = np.max(preds_proba)
    
    return result_index, confidence, class_names

# 3. SIDEBAR NAVIGATION 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# 4. PAGE: HOME
if app_mode == "Home":
    st.markdown("""
     ## **OCT Retinal Analysis Platform**

#### **Welcome to the Retinal OCT Analysis Platform**

**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**
OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

---

#### **Key Features of the Platform**

- **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

---

#### **Understanding Retinal Diseases through OCT**

1. **Choroidal Neovascularization (CNV)**
   - Neovascular membrane with subretinal fluid
   
2. **Diabetic Macular Edema (DME)**
   - Retinal thickening with intraretinal fluid
   
3. **Drusen (Early AMD)**
   - Presence of multiple drusen deposits

4. **Normal Retina**
   - Preserved foveal contour, absence of fluid or edema

---

#### **About the Dataset**

Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
- **Normal**
- **CNV**
- **DME**
- **Drusen**

Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

---

#### **Get Started**

- **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
- **Explore Results**: View categorized scans and detailed diagnostic insights.
- **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.

---

#### **Contact Us**

Have questions or need assistance? [Contact our support team](#) for more information on how to use the platform or integrate it into your clinical practice.

    """)

# 5. PAGE: ABOUT
elif app_mode == "About":
    st.header("About Project")
    st.markdown("""
                #### About Dataset
                Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living   patients. 
                Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
                (A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). 
                (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). 
                (Middle right) Multiple drusen (arrowheads) present in early AMD. 
                (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.
    
                ---
    
                #### Content
                The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). 
                There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).
    
                Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.
    
                Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients             from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates,             the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

                Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing exper- tise for            verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The            first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders             conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders             consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization             (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of             two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset             selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993             scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

                """)

# 6. PAGE: DISEASE IDENTIFICATION
elif app_mode == "Disease Identification":
    st.header("Medical Diagnostic Interface")
    
    uploaded_file = st.file_uploader("Upload Retinal OCT Scan (JPEG/PNG):", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # 2. Display Image
        st.image(uploaded_file, caption="Uploaded Scan", use_column_width=True)
        
        # 3. Predict Button
        if st.button("Run Diagnostic Analysis"):
            with st.spinner("Analyzing Retinal Layers..."):
                idx, conf, labels = hybrid_prediction(tmp_path)
                result_label = labels[idx]
            
            # Display Success
            st.success(f"**Analysis Complete:** Detected **{result_label}**")
            st.metric(label="Diagnostic Confidence", value=f"{conf:.2%}")
            
            # 4. Detailed Recommendation/Expander
            with st.expander("Clinical Insights & Recommendations"):
                if idx == 0: # CNV
                    st.info("Finding: CNV with subretinal fluid.")
                    st.markdown(cnv)
                elif idx == 1: # DME
                    st.info("Finding: DME with intraretinal fluid.")
                    st.markdown(dme)
                elif idx == 2: # DRUSEN
                    st.info("Finding: Drusen deposits detected.")
                    st.markdown(drusen)
                elif idx == 3: # NORMAL
                    st.success("Finding: Normal retina profile.")
                    st.markdown(normal)
            
            # Clean up temp file
            os.remove(tmp_path)
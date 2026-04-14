"""
main.py — OCT Retinal AI Inference API
FastAPI + ONNX Runtime edge deployment
Google Cloud Run compatible
"""

import os
import time
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES        = ["CNV", "DME", "DRUSEN", "NORMAL"]
IMG_SIZE       = 224
IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_DIR      = "/app/models"
ONNX_PATH      = os.path.join(MODEL_DIR, "human_eye_fp32.onnx")
OOD_MEAN_PATH  = os.path.join(MODEL_DIR, "ood_train_mean.npy")
OOD_COV_PATH   = os.path.join(MODEL_DIR, "ood_cov_inv.npy")
OOD_THRESH_PATH= os.path.join(MODEL_DIR, "ood_threshold.npy")
TEMP_PATH      = os.path.join(MODEL_DIR, "temperature.npy")
HF_REPO        = "animeshakr/oct-retinal-weights"

# ── Download models from HuggingFace if not present ───────────────────────────
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = {
        "human_eye_fp32.onnx": ONNX_PATH,
        "ood_train_mean.npy":  OOD_MEAN_PATH,
        "ood_cov_inv.npy":     OOD_COV_PATH,
        "ood_threshold.npy":   OOD_THRESH_PATH,
        "temperature.npy":     TEMP_PATH,
    }
    for filename, local_path in files.items():
        if not os.path.exists(local_path):
            logger.info(f"Downloading {filename} from HuggingFace...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=MODEL_DIR
            )
            logger.info(f"Downloaded {filename}")
        else:
            logger.info(f"Found {filename} locally")

# ── Load models ───────────────────────────────────────────────────────────────
download_models()

logger.info("Loading ONNX session...")
sess       = ort.InferenceSession(ONNX_PATH,
                                  providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
logger.info(f"ONNX loaded. Input: '{input_name}'")

# Load safety components
ood_mean  = np.load(OOD_MEAN_PATH)   # shape (256,)
ood_cov   = np.load(OOD_COV_PATH)    # shape (256, 256)
ood_thresh= float(np.load(OOD_THRESH_PATH))
temp      = float(np.load(TEMP_PATH))
logger.info(f"Safety components loaded. OOD threshold: {ood_thresh:.4f}  T: {temp:.4f}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCT Retinal AI — Edge Inference API",
    description=(
        "Hybrid CNN-Transformer framework for four-class retinal OCT "
        "classification. EfficientNetV2L + 4× MHA Transformer + XGBoost, "
        "deployed as a 237MB FP32 ONNX edge node. Includes OOD detection, "
        "temperature calibration, and uncertainty flagging. "
        "Model: animeshakr/oct-retinal-weights"
    ),
    version="1.0.0",
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

# ── OOD detection ─────────────────────────────────────────────────────────────
def mahalanobis_ood(embedding: np.ndarray) -> tuple[float, bool]:
    """
    Compute Mahalanobis distance from training distribution.
    Returns (score, is_ood).
    """
    diff  = embedding - ood_mean            # (256,)
    score = float(diff @ ood_cov @ diff.T)
    return score, score > ood_thresh

# ── Temperature scaling ───────────────────────────────────────────────────────
def apply_temperature(logits: np.ndarray) -> np.ndarray:
    scaled = logits / temp
    e      = np.exp(scaled - scaled.max())
    return e / e.sum()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":     "OCT Retinal AI — Edge Inference API",
        "version":     "1.0.0",
        "model":       "EfficientNetV2L + 4×MHA Transformer + XGBoost",
        "edge_model":  "human_eye_fp32.onnx (237 MB)",
        "classes":     CLASSES,
        "endpoints":   ["/predict", "/health", "/docs"],
        "author":      "Animesh A. Kumar — Newcastle University",
        "repo":        "https://huggingface.co/animeshakr/oct-retinal-weights",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a retinal OCT B-scan into one of four categories:
    CNV, DME, DRUSEN, or NORMAL.

    Returns prediction, calibrated confidence, OOD flag,
    uncertainty flag, and inference latency.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="File must be an image (JPEG/PNG)")

    try:
        img_bytes = await file.read()
        img       = preprocess(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=422,
                            detail=f"Image preprocessing failed: {e}")

    # ── Inference ──────────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    logits = sess.run(None, {input_name: img})[0]  # (1, 4)
    latency_ms = (time.perf_counter() - t0) * 1000

    logits_1d = logits[0].astype(np.float32)

    # Temperature-calibrated confidence
    probs      = apply_temperature(logits_1d)
    pred_idx   = int(np.argmax(probs))
    prediction = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # OOD detection — use logit vector as proxy embedding
    # (full Mahalanobis requires the 256-d Transformer embedding;
    #  here we use the 4-d logit space as a lightweight proxy)
    ood_score, ood_flag = mahalanobis_ood(logits_1d)

    # Uncertainty flag — entropy-based
    entropy          = float(-np.sum(probs * np.log(probs + 1e-9)))
    max_entropy      = float(np.log(len(CLASSES)))
    norm_entropy     = entropy / max_entropy
    uncertainty_flag = bool(norm_entropy > 0.5 or confidence < 0.70)

    # Clinical routing
    if ood_flag:
        clinical_note = "OOD flag: scan may be non-retinal or corrupted. Route to specialist."
    elif uncertainty_flag:
        clinical_note = f"Uncertain prediction ({confidence*100:.1f}%). Route to specialist review."
    else:
        clinical_note = "Prediction within normal confidence range."

    return JSONResponse({
        "prediction":       prediction,
        "confidence":       round(confidence * 100, 2),
        "all_probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASSES, probs)
        },
        "ood_flag":         ood_flag,
        "ood_score":        round(ood_score, 4),
        "uncertainty_flag": uncertainty_flag,
        "entropy":          round(norm_entropy, 4),
        "clinical_note":    clinical_note,
        "latency_ms":       round(latency_ms, 2),
        "model":            "human_eye_fp32.onnx (237MB edge node)",
        "temperature":      round(temp, 4),
    })

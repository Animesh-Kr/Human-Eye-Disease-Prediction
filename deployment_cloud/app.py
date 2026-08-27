"""
OCT Retinal Disease Classification Dashboard
EfficientNetV2L + 4× MHA + XGBoost Hybrid
Unified Cloud Deployment for Hugging Face Spaces
"""

import os
import json
import base64
import io
import math
import warnings
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from PIL import Image
from pathlib import Path

import tensorflow as tf
import xgboost as xgb
import albumentations as A
from scipy.spatial.distance import mahalanobis

# ── Audit fixes: MC Dropout and Grad-CAM correctness ─────────────────────────
# model(x, training=True) enables dropout but ALSO puts every BatchNormalization
# layer into training mode, so each BN normalises by the statistics of the current
# batch -- and that batch is n identical copies of one image, not the moving
# averages learned over the training set. EfficientNetV2L is saturated with BN.
# _mc_model() keeps BatchNorm in inference mode and makes only dropout stochastic.
_MC_CACHE = {}


def _mc_model(base):
    """Clone `base` with every Dropout swapped for an always-sampling Dropout.

    TensorFlow is imported locally: app.py loads it lazily inside a cached loader,
    so there is no module-level `tf` to rely on here.
    """
    import tensorflow as tf

    key = id(base)
    if key not in _MC_CACHE:
        class _MCDropout(tf.keras.layers.Dropout):
            def call(self, inputs, training=None):
                return super().call(inputs, training=True)

        def _swap(layer):
            if type(layer) is tf.keras.layers.Dropout:
                return _MCDropout(layer.rate, noise_shape=layer.noise_shape,
                                  seed=layer.seed, name=layer.name)
            return layer.__class__.from_config(layer.get_config())

        clone = tf.keras.models.clone_model(base, clone_function=_swap)
        clone.set_weights(base.get_weights())
        _MC_CACHE[key] = clone
    return _MC_CACHE[key]


def _softmax_dense(base, name='disease_output'):
    """The final softmax Dense layer, so Grad-CAM can reach its pre-activation."""
    import tensorflow as tf

    try:
        return base.get_layer(name)
    except ValueError:
        return next(l for l in reversed(base.layers)
                    if isinstance(l, tf.keras.layers.Dense))
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings('ignore')

# Must be FIRST Streamlit command
st.set_page_config(
    page_title="OCT Retinal AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THE MASTER PATCH: Fixes Keras 3 "quantization_config" crash on Hugging Face
# =============================================================================
if not hasattr(tf.keras.layers.Layer, "_is_patched"):
    original_layer_init = tf.keras.layers.Layer.__init__
    def patched_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) 
        original_layer_init(self, *args, **kwargs)
    tf.keras.layers.Layer.__init__ = patched_layer_init
    tf.keras.layers.Layer._is_patched = True

# =============================================================================
# PATHS & CONFIGURATION
# =============================================================================
BASE_DIR   = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CLASS_COLORS = {'CNV': '#E74C3C', 'DME': '#3498DB', 'DRUSEN': '#9B59B6', 'NORMAL': '#2ECC71'}
IMG_SIZE    = 224
N_MC_PASSES = 20
UNC_THRESH  = 0.15

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0a0e1a; }
  [data-testid="stSidebar"] * { color: #e8eaf0 !important; }
  [data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a3048;
    border-radius: 10px;
    padding: 12px;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 16px; font-size: 13px; }
  .clinical-warning {
    background: #3d1f00; border-left: 4px solid #F39C12;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 13px;
  }
  .clinical-safe {
    background: #0a2a12; border-left: 4px solid #2ECC71;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 13px;
  }
  .clinical-danger {
    background: #2a0a0a; border-left: 4px solid #E74C3C;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 13px;
  }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# KERAS CUSTOM OBJECTS
# =============================================================================
class _DummyFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'gamma': self.gamma, 'alpha': self.alpha})
        return cfg

@tf.keras.utils.register_keras_serializable()
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len=49, proj_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.seq_len  = seq_len
        self.proj_dim = proj_dim
        self.pos_embeddings = self.add_weight(
            shape=(1, seq_len, proj_dim),
            initializer="zeros",
            trainable=True,
            name="pos_embedding"
        )
    def call(self, inputs):
        return inputs + self.pos_embeddings
    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "proj_dim": self.proj_dim})
        return config

VAL_AUG = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# =============================================================================
# RESOURCE LOADING
# =============================================================================
@st.cache_resource
def load_local_models():
    model = tf.keras.models.load_model(
        str(MODELS_DIR / 'Final_CNN_Transformer.keras'),
        custom_objects={
            'FocalLoss': _DummyFocalLoss,
            'LearnablePositionalEncoding': LearnablePositionalEncoding,
            'Dense': tf.keras.layers.Dense
        },
        compile=False
    )
    feat_ext = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('feature_extraction_layer').output,
        name='feat_ext',
    )
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / 'Final_XGBoost_Hybrid.json'))
    
    train_mean    = np.load(str(MODELS_DIR / 'ood_train_mean.npy'))
    cov_inv       = np.load(str(MODELS_DIR / 'ood_cov_inv.npy'))
    ood_threshold = float(np.load(str(MODELS_DIR / 'ood_threshold.npy'))[0])
    temperature   = float(np.load(str(MODELS_DIR / 'temperature.npy'))[0])
    
    return model, feat_ext, xgb_model, train_mean, cov_inv, ood_threshold, temperature

model, feat_ext, xgb_model, train_mean, cov_inv, ood_threshold, temperature = load_local_models()

# =============================================================================
# INFERENCE HELPERS
# =============================================================================
def apply_clahe(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)

def apply_temperature_scaling(probs, T):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    scaled    = log_probs / T
    shifted   = scaled - scaled.max(axis=1, keepdims=True)
    exp_s     = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)

def run_ood_check(features):
    dist = mahalanobis(features.flatten(), train_mean, cov_inv)
    return bool(dist >= ood_threshold), float(dist)

def run_mc_dropout(x_input, n_passes=N_MC_PASSES):
    x_batch = np.repeat(x_input, n_passes, axis=0)
    # training=False: BatchNorm keeps its moving averages, _MCDropout still samples
    preds   = _mc_model(model)(x_batch, training=False).numpy()
    return preds.mean(axis=0), preds.std(axis=0)

def generate_gradcam(x_input, last_conv_layer='top_activation'):
    try:
        conv_layer = model.get_layer(last_conv_layer)
    except ValueError:
        conv_layer = next((l for l in model.layers if 'activation' in l.name and len(l.output_shape) == 4), None)
        if conv_layer is None: return np.zeros((7, 7))
        
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, _softmax_dense(model).input])
    with tf.GradientTape() as tape:
        # PRE-SOFTMAX logit, not the probability -- see Selvaraju et al. (2017).
        # Differentiating the softmax scales gradients by p(1-p), ~1e-3 at this
        # model's confidences, which washes the attribution out entirely.
        conv_out, penult = grad_model(x_input, training=False)
        _d = _softmax_dense(model)
        logits = tf.matmul(penult, _d.kernel) + _d.bias
        class_score = logits[:, tf.argmax(logits[0])]
    grads        = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = tf.nn.relu(tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis]))
    heatmap     /= (tf.reduce_max(heatmap) + 1e-8)
    res          = heatmap.numpy()
    if res.ndim == 0: res = res.reshape(1, 1)
    elif res.ndim == 1:
        s = max(1, int(np.sqrt(len(res))))
        res = res[:s*s].reshape(s, s)
    return res

def overlay_gradcam(original_rgb, heatmap):
    h = np.array(heatmap, dtype=np.float32)
    if h.ndim < 2: h = h.reshape(1, 1)
    h_up  = cv2.resize(h, (original_rgb.shape[1], original_rgb.shape[0]))
    h_col = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * h_up), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(0.45 * h_col + 0.55 * original_rgb)

def full_predict(img_bgr):
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aug_img   = VAL_AUG(image=img_rgb)['image'].astype(np.float32)
    x_input   = np.expand_dims(aug_img, axis=0)
    
    features  = feat_ext.predict(x_input, verbose=0)
    is_ood, ood_score = run_ood_check(features[0])
    
    raw_probs = xgb_model.predict_proba(features)
    cal_probs = apply_temperature_scaling(raw_probs, temperature)
    pred_idx  = int(cal_probs.argmax(axis=1)[0])
    confidence= float(cal_probs[0, pred_idx])
    probs_dict= {CLASS_NAMES[i]: float(cal_probs[0, i]) for i in range(4)}
    
    mean_p, std_p = run_mc_dropout(x_input)
    uncertainty   = float(std_p.max())
    
    heatmap  = generate_gradcam(x_input)
    overlay  = overlay_gradcam(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), heatmap)
    clahe    = apply_clahe(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)))
    
    return {
        'prediction': CLASS_NAMES[pred_idx], 'confidence': confidence,
        'probs': probs_dict, 'uncertainty': uncertainty,
        'is_ood': is_ood, 'ood_score': ood_score,
        'ood_threshold': ood_threshold, 'overlay': overlay,
        'clahe': clahe, 'original_rgb': cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)),
    }

# =============================================================================
# UI COMPONENTS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_image_bytes(path_str: str) -> bytes:
    img = Image.open(path_str).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def plot_3d_architecture():
    nodes = [
        {"label": "OCT Input\n224×224×3",        "pos": (0, 0, 0),    "color": "#7F8C8D", "size": 12},
        {"label": "EfficientNetV2L Stem",         "pos": (0, 0, 1),    "color": "#E67E22", "size": 14},
        {"label": "CNN Blocks 1–5\n(frozen)",     "pos": (0, 0, 2),    "color": "#E67E22", "size": 22},
        {"label": "CNN Block 6+\n(fine-tuned)",   "pos": (0, 0, 3),    "color": "#F39C12", "size": 26},
        {"label": "Patch Projection\n1280→256-d", "pos": (0, 0, 4),    "color": "#3498DB", "size": 14},
        {"label": "Learnable PE\n49 tokens",      "pos": (0, 0, 4.7),  "color": "#2980B9", "size": 12},
        {"label": "MHA Block 1\n16 heads",        "pos": (-1.2, 1, 6), "color": "#1ABC9C", "size": 18},
        {"label": "MHA Block 2\n16 heads",        "pos": (1.2, 1, 6),  "color": "#1ABC9C", "size": 18},
        {"label": "MHA Block 3\n16 heads",        "pos": (-1.2,-1, 6), "color": "#1ABC9C", "size": 18},
        {"label": "MHA Block 4\n16 heads",        "pos": (1.2, -1, 6), "color": "#1ABC9C", "size": 18},
        {"label": "GlobalAvgPool1D\n256-d feat",  "pos": (0, 0, 7),    "color": "#9B59B6", "size": 16},
        {"label": "XGBoost\n300 trees depth=4",   "pos": (0, 0, 8),    "color": "#27AE60", "size": 24},
        {"label": "Temperature\nScaling T≈1.05",  "pos": (0.8, 0, 8.7),"color": "#F1C40F", "size": 12},
        {"label": "CNV / DME\nDRUSEN / NORMAL",   "pos": (0, 0, 9.5),  "color": "#E74C3C", "size": 18},
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,7),(5,8),(5,9),
        (6,10),(7,10),(8,10),(9,10),(10,11),(11,12),(12,13),
    ]
    x_n = [n["pos"][0] for n in nodes]
    y_n = [n["pos"][1] for n in nodes]
    z_n = [n["pos"][2] for n in nodes]
    xe, ye, ze = [], [], []
    for e in edges:
        xe += [nodes[e[0]]["pos"][0], nodes[e[1]]["pos"][0], None]
        ye += [nodes[e[0]]["pos"][1], nodes[e[1]]["pos"][1], None]
        ze += [nodes[e[0]]["pos"][2], nodes[e[1]]["pos"][2], None]
    fig = go.Figure(data=[
        go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color='#555', width=2), hoverinfo='none'),
        go.Scatter3d(x=x_n, y=y_n, z=z_n, mode='markers+text', text=[n["label"] for n in nodes],
                     textposition='top center', textfont=dict(size=9, color='white'),
                     marker=dict(size=[n["size"] for n in nodes], color=[n["color"] for n in nodes],
                                 line=dict(width=2, color='white'), opacity=0.9), hovertemplate='<b>%{text}</b><extra></extra>'),
    ])
    fig.update_layout(
        title=dict(text="Interactive 3D Architecture: EfficientNetV2L + 4× MHA + XGBoost", font=dict(size=14)),
        showlegend=False,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   bgcolor='rgba(10,14,26,1)', camera=dict(eye=dict(x=2, y=2, z=1.2))),
        margin=dict(l=0, r=0, b=0, t=50), height=580, paper_bgcolor='rgba(10,14,26,0)',
    )
    return fig

def confidence_gauge_html(conf: float, pred_class: str) -> str:
    color     = CLASS_COLORS.get(pred_class, '#2ECC71')
    pct       = round(conf * 100, 1)
    r         = 70
    semi_circ = math.pi * r
    filled    = (pct / 100.0) * semi_circ
    gap       = semi_circ - filled
    dash      = f"{filled:.2f} {gap + semi_circ:.2f}"
    return f"""
<div style="text-align:center; padding:8px 0;">
  <svg viewBox="0 0 200 115" width="100%" style="max-width:280px; display:block; margin:0 auto;">
    <path d="M 15 100 A 70 70 0 0 1 185 100" fill="none" stroke="#2a3048" stroke-width="16" stroke-linecap="round"/>
    <path d="M 15 100 A 70 70 0 0 1 185 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
          stroke-dasharray="{dash}" pathLength="{semi_circ:.2f}"/>
    <text x="100" y="90" text-anchor="middle" font-size="28" font-weight="bold" fill="{color}">{pct}%</text>
    <text x="100" y="110" text-anchor="middle" font-size="11" fill="#8892a4">{pred_class}</text>
  </svg>
  <div style="color:#8892a4; font-size:11px; margin-top:2px;">Calibrated confidence</div>
</div>"""

def probs_bar_html(probs: dict) -> str:
    rows = ""
    for name, val in probs.items():
        color = CLASS_COLORS.get(name, "#888")
        pct   = round(val * 100, 1)
        rows += f"""
    <div style="display:flex;align-items:center;gap:8px;margin:5px 0;">
      <div style="width:55px;font-size:12px;color:#e8eaf0;text-align:right;flex-shrink:0;">{name}</div>
      <div style="flex:1;background:#1a1f2e;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:{min(pct,100):.1f}%;background:{color};height:100%;border-radius:4px;"></div>
      </div>
      <div style="width:42px;font-size:12px;color:{color};font-weight:600;flex-shrink:0;">{pct:.1f}%</div>
    </div>"""
    return f'<div style="padding:4px 0;">{rows}</div>'

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 👁️ OCT Retinal AI")
    st.markdown("**CNN-Transformer Hybrid**")
    st.markdown("---")

    st.success("☁️ **Cloud Inference Mode**\nLive prediction via Full Precision Pipeline.")
    uploaded_file = st.file_uploader("Upload OCT scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown("""
**Model specs:**
- Backbone: EfficientNetV2L
- Attention: 4× MHA (16 heads)
- Head: XGBoost (300 trees)
- Safety: Mahalanobis OOD
- Calibration: Temperature scaling

**Test set results:**
- Accuracy: 95.43% ± 0.27%
- Macro AUC: 0.9941 ± 0.0006
- ECE (cal): 0.0024 ± 0.0005
    """)
    st.markdown("---")

    with st.expander("⚙️ How the pipeline works"):
        st.markdown("""
1. **OOD check** — Mahalanobis distance against training distribution (threshold: 97th pct)
2. **Feature extraction** — EfficientNetV2L block6+ → 256-d vector
3. **XGBoost prediction** — 300 trees on extracted features
4. **Temperature calibration** — T≈1.05 softens overconfident probabilities
5. **Grad-CAM** — gradient-weighted activations on `top_activation` layer
6. **MC Dropout** — 20 stochastic passes estimate epistemic uncertainty
        """)

    st.markdown("---")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction)")
    st.caption("Kermany OCT Dataset · 84k scans · 4 classes")
    st.caption("Newcastle MSc 2025–26")

# =============================================================================
# HEADER
# =============================================================================
st.title("👁️  Retinal Disease Classification")
st.markdown("**EfficientNetV2L + 4× Multi-Head Attention + XGBoost** · Temperature-calibrated · OOD-guarded · MC Dropout uncertainty")

st.markdown("""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1rem;">
  <div style="background:#1a1f2e;border:1px solid #2a3048;border-radius:10px;padding:12px;">
    <div style="font-size:12px;color:#8892a4;">Accuracy</div>
    <div style="font-size:22px;font-weight:600;color:#e8eaf0;">95.43%</div>
    <div style="font-size:11px;color:#2ECC71;">5-seed mean ± 0.27%</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2a3048;border-radius:10px;padding:12px;">
    <div style="font-size:12px;color:#8892a4;">Macro AUC</div>
    <div style="font-size:22px;font-weight:600;color:#e8eaf0;">0.9941</div>
    <div style="font-size:11px;color:#2ECC71;">± 0.0006</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2a3048;border-radius:10px;padding:12px;">
    <div style="font-size:12px;color:#8892a4;">Drusen F1</div>
    <div style="font-size:22px;font-weight:600;color:#e8eaf0;">0.8436</div>
    <div style="font-size:11px;color:#F39C12;">hardest class</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2a3048;border-radius:10px;padding:12px;">
    <div style="font-size:12px;color:#8892a4;">ECE (cal)</div>
    <div style="font-size:22px;font-weight:600;color:#e8eaf0;">0.0024</div>
    <div style="font-size:11px;color:#2ECC71;">well-calibrated</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2a3048;border-radius:10px;padding:12px;">
    <div style="font-size:12px;color:#8892a4;">McNemar p</div>
    <div style="font-size:22px;font-weight:600;color:#e8eaf0;">0.0001</div>
    <div style="font-size:11px;color:#2ECC71;">all 5 seeds sig.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺  Clinical Workspace",
    "🧠  3D Architecture",
    "🔍  Explainability",
    "🌌  Feature Space",
    "📊  Phase 6 Validation",
])

# ── TAB 1: CLINICAL WORKSPACE ─────────────────────────────────────────────────
with tab1:
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Could not decode image.")
        else:
            with st.spinner("Processing through the AI pipeline..."):
                result = full_predict(img_bgr)

            col_img, col_result = st.columns([1, 1])
            with col_img:
                st.subheader("OCT Scan")
                img_tabs = st.tabs(["Original", "CLAHE Enhanced", "Grad-CAM"])
                with img_tabs[0]:
                    st.image(result['original_rgb'], use_container_width=True)
                with img_tabs[1]:
                    st.image(result['clahe'], use_container_width=True)
                    st.caption("CLAHE — enhances retinal layer contrast")
                with img_tabs[2]:
                    st.image(result['overlay'], use_container_width=True)
                    st.caption("Red = highest influence on prediction")

            with col_result:
                st.subheader("Prediction")
                st.markdown(confidence_gauge_html(result['confidence'], result['prediction']), unsafe_allow_html=True)
                st.markdown(probs_bar_html(result['probs']), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                if result['is_ood']:
                    st.markdown(
                        f'<div class="clinical-danger">⛔ <b>OOD DETECTED</b> — '
                        f'Score {result["ood_score"]:.2f} > threshold {result["ood_threshold"]:.2f}.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="clinical-safe">✅ <b>In-distribution</b> — '
                        f'Score {result["ood_score"]:.2f} (threshold {result["ood_threshold"]:.2f})</div>',
                        unsafe_allow_html=True)

                unc = result['uncertainty']
                if unc > UNC_THRESH:
                    st.markdown(
                        f'<div class="clinical-warning">⚠️ <b>High uncertainty ({unc:.3f})</b> — Specialist review recommended.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="clinical-safe">✅ <b>Low uncertainty ({unc:.3f})</b> — {N_MC_PASSES} MC passes agree.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown("### 👈 Upload an OCT scan in the sidebar to begin")
        st.markdown("""
**Supported inputs:** OCT B-scan images (.jpg / .png)
**Classes:** CNV · DME · DRUSEN · NORMAL
**Pipeline:** OOD check → Feature extraction → XGBoost → Temperature calibration → Grad-CAM → MC Dropout
        """)

# ── TAB 2: 3D ARCHITECTURE ────────────────────────────────────────────────────
with tab2:
    st.subheader("Hybrid CNN-Transformer Architecture (Interactive 3D)")
    st.markdown("Drag to rotate · Scroll to zoom · Hover nodes for details. Node size ∝ parameter count.")
    st.plotly_chart(plot_3d_architecture(), use_container_width=True, key="arch_3d")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Architecture highlights:**
- EfficientNetV2L backbone (118.5M params, pretrained ImageNet)
- Blocks 1–5 frozen in Phase A; Block 6+ fine-tuned in Phase B
- Patch reshape: 7×7×1280 → 49 tokens · Linear projection: 1280→256-d
- Learnable positional encoding (not sinusoidal)
- 4× Transformer encoder blocks, each with 16-head MHA
- Feed-forward expansion: 4× (256→1024→256)
- Global Average Pooling → 256-d feature vector
- XGBoost hybrid head (300 trees, max_depth=4)
        """)
    with c2:
        st.markdown("""
**Training protocol:**
- Phase A: head only, 5 epochs, Adam lr=1.59e-4
- Phase B: block6+ unfrozen, 20 epochs, WarmupCosineDecay
- Loss: Focal Loss (γ=1.36, per-class α)
- Augmentation: Albumentations + CutMix (batch=512)
- HPO: Optuna TPE, 10 trials
- Calibration: Temperature scaling (T≈1.05)
- Safety: Mahalanobis OOD (97th percentile threshold)
- Uncertainty: MC Dropout (20 stochastic passes)
        """)

# ── TAB 3: EXPLAINABILITY ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Global Explainability — Grad-CAM · SHAP · Attention Maps")
    ex_tabs = st.tabs(["Grad-CAM Panel", "SHAP Summary", "Attention Heads"])
    with ex_tabs[0]:
        try:
            st.image(load_image_bytes(str(ASSETS_DIR / "gradcam_panel.png")), caption="Grad-CAM: one representative sample per class.", use_container_width=True)
        except: pass
    with ex_tabs[1]:
        try:
            st.image(load_image_bytes(str(ASSETS_DIR / "shap_summary.png")), caption="SHAP: top transformer features driving XGBoost.", use_container_width=True)
        except: pass
    with ex_tabs[2]:
        for cls in CLASS_NAMES:
            try:
                st.image(load_image_bytes(str(ASSETS_DIR / f"attention_heads_{cls.lower()}.png")), caption=f"Attention heads — {cls}", use_container_width=True)
            except: pass

# ── TAB 4: FEATURE SPACE ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Feature Space · Uncertainty Landscape")
    fs_tabs = st.tabs(["UMAP 3D (interactive)", "Uncertainty Landscape", "UMAP 2D"])
    with fs_tabs[0]:
        try:
            with open(str(ASSETS_DIR / "umap_3d_features.html"), "r", encoding="utf-8") as f:
                components.html(f.read(), height=620, scrolling=True)
        except: pass
    with fs_tabs[1]:
        try:
            with open(str(ASSETS_DIR / "uncertainty_landscape.html"), "r", encoding="utf-8") as f:
                components.html(f.read(), height=620, scrolling=True)
        except: 
            try: st.image(load_image_bytes(str(ASSETS_DIR / "uncertainty_landscape.png")), use_container_width=True)
            except: pass
    with fs_tabs[2]:
        try:
            st.image(load_image_bytes(str(ASSETS_DIR / "umap_2d_features.png")), use_container_width=True)
        except: pass

# ── TAB 5: PHASE 6 VALIDATION ─────────────────────────────────────────────────
with tab5:
    st.subheader("Multi-Seed Statistical Validation (n=5 seeds)")
    st.markdown("Hyperparameters fixed to Optuna best from seed=42. Only weight initialisation and data shuffle vary.")
    try:
        agg = pd.read_csv(str(ASSETS_DIR / "multiseed_aggregate.csv"), index_col=0)
        if 'Mean' not in agg.columns: agg.columns = ['Mean', 'Std']
        display_metrics = ['accuracy', 'macro_f1', 'drusen_f1', 'macro_auc', 'ece_cal', 'mcnemar_p']
        labels = {'accuracy': 'Accuracy', 'macro_f1': 'Macro F1', 'drusen_f1': 'Drusen F1 (minority class)', 'macro_auc': 'Macro AUC-ROC', 'ece_cal': 'ECE (calibrated)', 'mcnemar_p': 'McNemar p-value'}
        pub_df = pd.DataFrame({'Metric': [labels.get(m, m) for m in display_metrics if m in agg.index], 'Mean ± Std': [f"{agg.loc[m,'Mean']:.4f} ± {agg.loc[m,'Std']:.4f}" for m in display_metrics if m in agg.index]})
        st.dataframe(pub_df, use_container_width=True, hide_index=True)
    except: pass
    st.markdown("---")
    try:
        st.image(load_image_bytes(str(ASSETS_DIR / "multiseed_violin.png")), caption="Metric distributions across 5 seeds.", use_container_width=True)
    except: pass
    st.markdown("---")
    st.subheader("Dataset — Class Distribution")
    try:
        st.image(load_image_bytes(str(ASSETS_DIR / "class_distribution.png")), caption="Kermany OCT dataset class imbalance.", use_container_width=True)
    except: pass
    st.markdown("---")
    st.subheader("Clinical Significance")
    st.markdown("""
**Why this problem matters:**

**CNV (Choroidal Neovascularisation)** — abnormal blood vessel growth beneath the retina, a leading cause of irreversible vision loss in wet AMD. Requires urgent anti-VEGF injection within days of detection.

**DME (Diabetic Macular Edema)** — fluid accumulation in the macula caused by diabetic retinopathy, affecting ~10% of diabetic patients. Early detection prevents progression to blindness.

**DRUSEN** — lipid deposits beneath the retinal pigment epithelium, the earliest biomarker of dry AMD. Detecting drusen early (before symptoms) is the critical clinical window for lifestyle intervention. This is the hardest class (F1=0.84) due to subtle appearance and 5.8× class imbalance.

**NORMAL** — healthy retina with no pathological features.

This model is designed to **triage**, not replace, ophthalmologists. The OOD detector and MC Dropout uncertainty flag scans requiring specialist review, ensuring the system fails safely.
    """)
    st.markdown("---")
    st.markdown("""
**Links:**
- 💻 [GitHub Repository](https://github.com/Animesh-Kr/Human-Eye-Disease-Prediction)
- 🤗 [HuggingFace Space](https://huggingface.co/spaces/animeshakr/oct-retinal-ai)
- 📊 Dataset: [Kermany et al., Cell 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
    """)
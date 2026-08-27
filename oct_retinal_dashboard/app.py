"""
OCT Retinal Disease Classification Dashboard
EfficientNetV2L + 4× MHA + XGBoost Hybrid

Two-tier deployment:
  Local (RTX 4060): full 2GB model, ~150ms inference
  HuggingFace Spaces: DEMO_MODE=true, precomputed JSON, instant load
"""

import os
import json
import base64
import io
import warnings
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from PIL import Image

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
    page_title   = "OCT Retinal AI",
    page_icon    = "👁️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CONFIGURATION
# =============================================================================
DEMO_MODE   = os.environ.get('DEMO_MODE', 'false').lower() == 'true'
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CLASS_COLORS= {'CNV': '#E74C3C', 'DME': '#3498DB', 'DRUSEN': '#9B59B6', 'NORMAL': '#2ECC71'}
IMG_SIZE    = 224
N_MC_PASSES = 20   # MC Dropout passes for uncertainty estimation
UNC_THRESH  = 0.15  # Refer to specialist above this

# =============================================================================
# CUSTOM CSS — clinical dark-accent theme
# =============================================================================
st.markdown("""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] { background: #0a0e1a; }
  [data-testid="stSidebar"] * { color: #e8eaf0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a3048;
    border-radius: 10px;
    padding: 12px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 13px;
  }

  /* Warning / success boxes */
  .clinical-warning {
    background: #3d1f00; border-left: 4px solid #F39C12;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 13px;
  }
  .clinical-safe {
    background: #0a2a12; border-left: 4px solid #2ECC71;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 13px;
  }
  .clinical-danger {
    background: #2a0a0a; border-left: 4px solid #E74C3C;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 13px;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# KERAS CUSTOM OBJECTS — local mode only
# =============================================================================
if not DEMO_MODE:
    import tensorflow as tf
    import keras
    import xgboost as xgb
    import albumentations as A
    from scipy.spatial.distance import mahalanobis

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

    @keras.saving.register_keras_serializable(package='Custom')
    class LearnablePositionalEncoding(keras.layers.Layer):
        def __init__(self, seq_len=49, proj_dim=512, **kwargs):
            super().__init__(**kwargs)
            self.seq_len  = seq_len
            self.proj_dim = proj_dim
        def build(self, input_shape):
            self.pos_embedding = self.add_weight(
                name='pos_embedding', shape=(1, self.seq_len, self.proj_dim),
                initializer='random_normal', trainable=True,
            )
            super().build(input_shape)
        def call(self, x):
            return x + tf.cast(self.pos_embedding, dtype=x.dtype)
        def get_config(self):
            cfg = super().get_config()
            cfg.update({'seq_len': self.seq_len, 'proj_dim': self.proj_dim})
            return cfg

    # Albumentations preprocessing — must match training exactly
    VAL_AUG = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# =============================================================================
# RESOURCE LOADING — cached, runs once
# =============================================================================
@st.cache_resource
def load_demo_data():
    try:
        with open('demo_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("demo_results.json not found. Run generate_demo.py first.")
        return {}


@st.cache_resource
def load_local_models():
    """Load full 2GB model into GPU VRAM — called once on startup."""
    st.info("Loading model into GPU VRAM... (one-time, ~20 seconds)")
    model = tf.keras.models.load_model(
        'models/Final_CNN_Transformer.keras',
        custom_objects={
            'FocalLoss'                   : _DummyFocalLoss,
            'LearnablePositionalEncoding' : LearnablePositionalEncoding,
        }
    )
    feat_ext = tf.keras.Model(
        inputs  = model.input,
        outputs = model.get_layer('feature_extraction_layer').output,
        name    = 'feat_ext',
    )
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/Final_XGBoost_Hybrid.json')

    train_mean    = np.load('models/ood_train_mean.npy')
    cov_inv       = np.load('models/ood_cov_inv.npy')
    ood_threshold = float(np.load('models/ood_threshold.npy')[0])
    temperature   = float(np.load('models/temperature.npy')[0])

    return model, feat_ext, xgb_model, train_mean, cov_inv, ood_threshold, temperature


if DEMO_MODE:
    demo_data = load_demo_data()
else:
    model, feat_ext, xgb_model, train_mean, cov_inv, ood_threshold, temperature = load_local_models()


# =============================================================================
# INFERENCE HELPERS — local mode
# =============================================================================
if not DEMO_MODE:

    def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
        """Resize + normalize using training-identical Albumentations pipeline."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return VAL_AUG(image=img_rgb)['image'].astype(np.float32)

    def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def apply_temperature_scaling(probs: np.ndarray, T: float) -> np.ndarray:
        log_probs = np.log(np.clip(probs, 1e-10, 1.0))
        scaled    = log_probs / T
        shifted   = scaled - scaled.max(axis=1, keepdims=True)
        exp_s     = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def run_ood_check(features: np.ndarray) -> tuple:
        dist = mahalanobis(features.flatten(), train_mean, cov_inv)
        return bool(dist >= ood_threshold), float(dist)

    def run_mc_dropout(x_input: np.ndarray, n_passes: int = N_MC_PASSES) -> tuple:
        """Batched MC Dropout — all passes in ONE GPU call."""
        x_batch = np.repeat(x_input, n_passes, axis=0)   # (n_passes, H, W, 3)
        # training=False: BatchNorm keeps its moving averages, _MCDropout still samples
        preds   = _mc_model(model)(x_batch, training=False).numpy()
        return preds.mean(axis=0), preds.std(axis=0)

    def generate_gradcam(x_input: np.ndarray,
                          last_conv_layer: str = 'top_activation') -> np.ndarray:
        """Grad-CAM heatmap. Returns 2D float array."""
        try:
            conv_layer = model.get_layer(last_conv_layer)
        except ValueError:
            # Fallback: find first layer with 'activation' in name
            conv_layer = next(
                (l for l in model.layers if 'activation' in l.name
                 and len(l.output_shape) == 4), None)
            if conv_layer is None:
                return np.zeros((7, 7))

        grad_model = tf.keras.Model(
            inputs=model.inputs, outputs=[conv_layer.output, _softmax_dense(model).input])
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
        heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.nn.relu(tf.squeeze(heatmap))
        heatmap     /= (tf.reduce_max(heatmap) + 1e-8)
        result       = heatmap.numpy()
        # Guarantee 2D
        if result.ndim == 0: result = result.reshape(1, 1)
        elif result.ndim == 1:
            s = max(1, int(np.sqrt(len(result))))
            result = result[:s*s].reshape(s, s)
        return result

    def overlay_gradcam(original_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        h = np.array(heatmap, dtype=np.float32)
        if h.ndim < 2: h = h.reshape(1, 1)
        h_up  = cv2.resize(h, (original_rgb.shape[1], original_rgb.shape[0]))
        h_col = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * h_up), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)
        return np.uint8(0.45 * h_col + 0.55 * original_rgb)

    def full_predict(img_bgr: np.ndarray) -> dict:
        """
        Complete inference pipeline:
        preprocess → OOD check → feature extract → XGB predict
        → temperature calibration → Grad-CAM → MC Dropout uncertainty
        """
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug_img   = VAL_AUG(image=img_rgb)['image'].astype(np.float32)
        x_input   = np.expand_dims(aug_img, axis=0)

        # Features
        features  = feat_ext.predict(x_input, verbose=0)

        # OOD
        is_ood, ood_score = run_ood_check(features[0])

        # XGBoost + temperature calibration
        raw_probs = xgb_model.predict_proba(features)
        cal_probs = apply_temperature_scaling(raw_probs, temperature)
        pred_idx  = int(cal_probs.argmax(axis=1)[0])
        confidence= float(cal_probs[0, pred_idx])
        probs_dict= {CLASS_NAMES[i]: float(cal_probs[0, i]) for i in range(4)}

        # MC Dropout uncertainty
        mean_p, std_p = run_mc_dropout(x_input)
        uncertainty   = float(std_p.max())

        # Grad-CAM
        heatmap  = generate_gradcam(x_input)
        overlay  = overlay_gradcam(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), heatmap)

        # CLAHE
        clahe    = apply_clahe(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)))

        return {
            'prediction'  : CLASS_NAMES[pred_idx],
            'confidence'  : confidence,
            'probs'       : probs_dict,
            'uncertainty' : uncertainty,
            'is_ood'      : is_ood,
            'ood_score'   : ood_score,
            'ood_threshold': ood_threshold,
            'overlay'     : overlay,
            'clahe'       : clahe,
            'original_rgb': cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)),
        }


# =============================================================================
# 3D ARCHITECTURE GRAPH
# =============================================================================
def plot_3d_architecture() -> go.Figure:
    nodes = [
        {"label": "OCT Input\n224×224×3",         "pos": (0, 0, 0),    "color": "#7F8C8D", "size": 12},
        {"label": "EfficientNetV2L Stem",          "pos": (0, 0, 1),    "color": "#E67E22", "size": 14},
        {"label": "CNN Blocks 1–5\n(frozen)",      "pos": (0, 0, 2),    "color": "#E67E22", "size": 22},
        {"label": "CNN Block 6+\n(fine-tuned)",    "pos": (0, 0, 3),    "color": "#F39C12", "size": 26},
        {"label": "Patch Projection\n1280→256-d",  "pos": (0, 0, 4),    "color": "#3498DB", "size": 14},
        {"label": "Learnable PE\n49 tokens",       "pos": (0, 0, 4.7),  "color": "#2980B9", "size": 12},
        {"label": "MHA Block 1\n16 heads",         "pos": (-1.2, 1,  6),"color": "#1ABC9C", "size": 18},
        {"label": "MHA Block 2\n16 heads",         "pos": (1.2,  1,  6),"color": "#1ABC9C", "size": 18},
        {"label": "MHA Block 3\n16 heads",         "pos": (-1.2, -1, 6),"color": "#1ABC9C", "size": 18},
        {"label": "MHA Block 4\n16 heads",         "pos": (1.2,  -1, 6),"color": "#1ABC9C", "size": 18},
        {"label": "GlobalAvgPool1D\n256-d feat",   "pos": (0, 0, 7),    "color": "#9B59B6", "size": 16},
        {"label": "XGBoost\n300 trees depth=4",    "pos": (0, 0, 8),    "color": "#27AE60", "size": 24},
        {"label": "Temperature\nScaling T=1.xx",   "pos": (0.8, 0, 8.7),"color": "#F1C40F", "size": 12},
        {"label": "CNV / DME\nDRUSEN / NORMAL",    "pos": (0, 0, 9.5),  "color": "#E74C3C", "size": 18},
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
        go.Scatter3d(x=xe, y=ye, z=ze, mode='lines',
                     line=dict(color='#555', width=2), hoverinfo='none'),
        go.Scatter3d(x=x_n, y=y_n, z=z_n,
                     mode='markers+text',
                     text=[n["label"] for n in nodes],
                     textposition='top center',
                     textfont=dict(size=9, color='white'),
                     marker=dict(
                         size=[n["size"] for n in nodes],
                         color=[n["color"] for n in nodes],
                         line=dict(width=2, color='white'),
                         opacity=0.9,
                     ),
                     hovertemplate='<b>%{text}</b><extra></extra>'),
    ])
    fig.update_layout(
        title=dict(text="Interactive 3D Architecture: EfficientNetV2L + 4× MHA + XGBoost",
                   font=dict(size=14)),
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False, title='Layer depth'),
            bgcolor='rgba(10,14,26,1)',
            camera=dict(eye=dict(x=2, y=2, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=580,
        paper_bgcolor='rgba(10,14,26,0)',
    )
    return fig


def confidence_gauge(conf: float, pred_class: str) -> go.Figure:
    color = CLASS_COLORS.get(pred_class, '#2ECC71')
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(conf * 100, 1),
        title = {'text': f"Prediction: {pred_class}", 'font': {'size': 16}},
        number= {'suffix': '%', 'font': {'size': 32}},
        gauge = {
            'axis'  : {'range': [0, 100], 'tickfont': {'size': 11}},
            'bar'   : {'color': color, 'thickness': 0.3},
            'steps' : [
                {'range': [0,  50],  'color': 'rgba(200,0,0,0.07)'},
                {'range': [50, 80],  'color': 'rgba(200,140,0,0.07)'},
                {'range': [80, 100], 'color': 'rgba(0,180,0,0.07)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3},
                          'thickness': 0.75, 'value': conf * 100},
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig


def probs_bar_chart(probs: dict) -> go.Figure:
    names   = list(probs.keys())
    values  = [round(v * 100, 2) for v in probs.values()]
    colors  = [CLASS_COLORS[n] for n in names]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker_color=colors, marker_opacity=0.85,
        text=[f"{v:.1f}%" for v in values], textposition='outside',
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110], visible=False),
        yaxis=dict(tickfont=dict(size=12, color='white')),
        height=160, margin=dict(l=10, r=50, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/400px-Image_created_with_a_mobile_phone.png",
             width=40, caption="") if False else None   # placeholder removed

    st.markdown("## 👁️ OCT Retinal AI")
    st.markdown("**CNN-Transformer Hybrid**")
    st.markdown("---")

    if DEMO_MODE:
        st.info("☁️ **Cloud Demo Mode**\nPrecomputed results — instant load")
        if demo_data:
            sample_key = st.selectbox(
                "Select sample scan:",
                options=list(demo_data.keys()),
                format_func=lambda x: x.replace('_', ' ').title(),
            )
        else:
            sample_key  = None
        uploaded_file = None
    else:
        st.success("🚀 **Local GPU Mode**\nRTX 4060 · ~150ms inference")
        uploaded_file = st.file_uploader(
            "Upload OCT scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
        sample_key    = None

    st.markdown("---")
    st.markdown("""
**Model specs:**
- Backbone: EfficientNetV2L
- Attention: 4× MHA (16 heads)
- Head: XGBoost (300 trees)
- Safety: Mahalanobis OOD
- Calibration: Temperature scaling

**Test set results:**
- Accuracy: 95.9%
- Macro AUC: 0.9947
- ECE (cal): 0.0017
    """)

    st.markdown("---")
    st.caption("Kermany OCT Dataset · 84k scans · 4 classes")
    st.caption("Newcastle MSc 2025–26")


# =============================================================================
# HEADER
# =============================================================================
st.title("👁️  Retinal Disease Classification")
st.markdown(
    "**EfficientNetV2L + 4× Multi-Head Attention + XGBoost** · "
    "Temperature-calibrated · OOD-guarded · MC Dropout uncertainty"
)

# Key metrics strip
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy",    "95.9%",  delta="vs Kermany 96.6%")
m2.metric("Macro AUC",   "0.9947", delta="+CI [0.9937–0.9958]")
m3.metric("Drusen F1",   "0.8553", delta="hardest class")
m4.metric("ECE (cal)",   "0.0017", delta="well-calibrated")
m5.metric("McNemar p",   "0.00066",delta="vs LR baseline")

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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLINICAL WORKSPACE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # ── DEMO MODE ────────────────────────────────────────────────────────────
    if DEMO_MODE and sample_key and sample_key in demo_data:
        d = demo_data[sample_key]

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.subheader("OCT Scan")
            orig_bytes = base64.b64decode(d['original_b64'])
            gc_bytes   = base64.b64decode(d['gradcam_b64'])
            st.image(orig_bytes, caption="Original scan", use_container_width=True)
            st.image(gc_bytes,   caption="Grad-CAM overlay", use_container_width=True)

        with col_result:
            st.subheader("Prediction")
            st.plotly_chart(
                confidence_gauge(d['confidence'], d['prediction']),
                use_container_width=True)
            st.plotly_chart(
                probs_bar_chart(d['probs']),
                use_container_width=True)

            # OOD status
            if d.get('is_ood'):
                st.markdown(
                    '<div class="clinical-danger">⛔ <b>OOD DETECTED</b> — '
                    'This scan deviates significantly from the training distribution. '
                    'Do not rely on this prediction.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="clinical-safe">✅ <b>In-distribution scan</b> — '
                    f'Mahalanobis score: {d.get("ood_score", 0):.2f} '
                    f'(threshold: {d.get("ood_threshold", 26.55):.2f})</div>',
                    unsafe_allow_html=True)

            # Uncertainty
            unc = d.get('uncertainty', 0)
            if unc > UNC_THRESH:
                st.markdown(
                    f'<div class="clinical-warning">⚠️ <b>High epistemic uncertainty '
                    f'({unc:.3f})</b> — Manual specialist review recommended.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="clinical-safe">✅ <b>Low uncertainty ({unc:.3f})</b> '
                    f'— {N_MC_PASSES} MC Dropout passes agree.</div>',
                    unsafe_allow_html=True)

    # ── LOCAL GPU MODE ────────────────────────────────────────────────────────
    elif not DEMO_MODE and uploaded_file is not None:
        # Decode uploaded image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not decode image. Please upload a valid JPG or PNG.")
        else:
            with st.spinner("Running inference on GPU..."):
                result = full_predict(img_bgr)

            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.subheader("OCT Scan")
                img_tab = st.tabs(["Original", "CLAHE Enhanced", "Grad-CAM"])
                with img_tab[0]:
                    st.image(result['original_rgb'], use_container_width=True)
                with img_tab[1]:
                    st.image(result['clahe'], use_container_width=True)
                    st.caption("CLAHE — enhances retinal layer contrast")
                with img_tab[2]:
                    st.image(result['overlay'], use_container_width=True)
                    st.caption("Red = highest influence on prediction")

            with col_result:
                st.subheader("Prediction")
                st.plotly_chart(
                    confidence_gauge(result['confidence'], result['prediction']),
                    use_container_width=True)
                st.plotly_chart(
                    probs_bar_chart(result['probs']),
                    use_container_width=True)

                # OOD
                if result['is_ood']:
                    st.markdown(
                        f'<div class="clinical-danger">⛔ <b>OOD DETECTED</b> — '
                        f'Score {result["ood_score"]:.2f} > threshold '
                        f'{result["ood_threshold"]:.2f}. '
                        f'Do not rely on this prediction.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="clinical-safe">✅ <b>In-distribution</b> — '
                        f'Score {result["ood_score"]:.2f} '
                        f'(threshold {result["ood_threshold"]:.2f})</div>',
                        unsafe_allow_html=True)

                # Uncertainty
                unc = result['uncertainty']
                if unc > UNC_THRESH:
                    st.markdown(
                        f'<div class="clinical-warning">⚠️ <b>High uncertainty '
                        f'({unc:.3f})</b> — Specialist review recommended.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="clinical-safe">✅ <b>Low uncertainty ({unc:.3f})</b>'
                        f' — {N_MC_PASSES} MC passes agree.</div>',
                        unsafe_allow_html=True)

    else:
        st.markdown("### 👈 Upload a scan or select a sample to begin")
        st.markdown("""
        **Supported inputs:** OCT B-scan images (.jpg / .png)  
        **Classes:** CNV · DME · DRUSEN · NORMAL  
        **Pipeline:** OOD check → Feature extraction → XGBoost → Temperature calibration → Grad-CAM → MC Dropout
        """)
        st.info("**Sample scans:** A set of representative scans is available "
                "in the sidebar (demo mode) or you can upload your own.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — 3D ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Hybrid CNN-Transformer Architecture (Interactive 3D)")
    st.markdown(
        "Drag to rotate · Scroll to zoom · Hover nodes for details. "
        "Node size ∝ parameter count.")
    st.plotly_chart(plot_3d_architecture(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Architecture highlights:**
- **EfficientNetV2L** backbone (118.5M params, pretrained ImageNet)
- Blocks 1–5 frozen during Phase A; Block 6+ fine-tuned in Phase B
- Patch reshape: 7×7×1280 → 49 tokens
- Linear projection: 1280 → 256-d
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
- Calibration: Temperature scaling (validation NLL)
- Safety: Mahalanobis OOD (97th percentile threshold)
- Uncertainty: MC Dropout (20 stochastic passes)
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Global Explainability — Grad-CAM · SHAP · Attention Maps")

    ex_tabs = st.tabs(["Grad-CAM Panel", "SHAP Summary", "Attention Heads"])

    with ex_tabs[0]:
        try:
            st.image("assets/gradcam_panel.png",
                     caption="Grad-CAM: one representative sample per class. "
                              "Red = highest influence.",
                     use_container_width=True)
        except Exception:
            st.info("Place gradcam_panel.png in assets/")

    with ex_tabs[1]:
        try:
            st.image("assets/shap_summary.png",
                     caption="SHAP: top transformer features driving XGBoost "
                              "predictions per class.",
                     use_container_width=True)
        except Exception:
            st.info("Place shap_summary.png in assets/")

    with ex_tabs[2]:
        attn_cols = st.columns(2)
        for idx, cls in enumerate(CLASS_NAMES):
            with attn_cols[idx % 2]:
                try:
                    st.image(f"assets/attention_heads_{cls.lower()}.png",
                             caption=f"Attention heads — {cls} "
                                      "(each head attends to different retinal regions)",
                             use_container_width=True)
                except Exception:
                    st.info(f"Place attention_heads_{cls.lower()}.png in assets/")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FEATURE SPACE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Feature Space · Uncertainty Landscape")

    fs_tabs = st.tabs(["UMAP 3D (interactive)", "Uncertainty Landscape (interactive)",
                        "UMAP 2D (static)"])

    with fs_tabs[0]:
        try:
            with open("assets/umap_3d_features.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=600, scrolling=False)
        except FileNotFoundError:
            st.info("Place umap_3d_features.html in assets/ to embed the interactive 3D UMAP.")

    with fs_tabs[1]:
        try:
            with open("assets/uncertainty_landscape.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=600, scrolling=False)
        except FileNotFoundError:
            try:
                st.image("assets/uncertainty_landscape.png",
                         caption="Peaks = decision boundaries · Valleys = confident predictions",
                         use_container_width=True)
            except Exception:
                st.info("Place uncertainty_landscape.html or .png in assets/")

    with fs_tabs[2]:
        try:
            st.image("assets/umap_2d_features.png",
                     caption="Test set feature space (UMAP 2D). "
                              "Well-separated clusters confirm discriminative learning.",
                     use_container_width=True)
        except Exception:
            st.info("Place umap_2d_features.png in assets/")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PHASE 6 VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Multi-Seed Statistical Validation (n=5 seeds)")
    st.markdown(
        "Hyperparameters fixed to Optuna best from seed=42. "
        "Only weight initialisation and data shuffle vary."
    )

    try:
        agg = pd.read_csv("assets/multiseed_aggregate.csv", index_col=0)
        # Force correct columns if needed
        if 'Mean' not in agg.columns:
            agg.columns = ['Mean', 'Std']

        # Publication-ready metrics
        display_metrics = ['accuracy', 'macro_f1', 'drusen_f1',
                           'macro_auc', 'ece_cal', 'mcnemar_p']
        labels = {
            'accuracy'  : 'Accuracy',
            'macro_f1'  : 'Macro F1',
            'drusen_f1' : 'Drusen F1 (minority class)',
            'macro_auc' : 'Macro AUC-ROC',
            'ece_cal'   : 'ECE (calibrated)',
            'mcnemar_p' : 'McNemar p-value',
        }

        pub_df = pd.DataFrame({
            'Metric'    : [labels.get(m, m) for m in display_metrics if m in agg.index],
            'Mean ± Std': [f"{agg.loc[m,'Mean']:.4f} ± {agg.loc[m,'Std']:.4f}"
                           for m in display_metrics if m in agg.index],
        })
        st.dataframe(pub_df, use_container_width=True, hide_index=True)

        st.markdown("---")

    except FileNotFoundError:
        st.info("Phase 6 will complete overnight. "
                "Place multiseed_aggregate.csv in assets/ when ready.")

    # Violin plot
    try:
        st.image("assets/multiseed_violin.png",
                 caption="Metric distributions across 5 seeds. "
                          "Narrow violins confirm stable training.",
                 use_container_width=True)
    except Exception:
        st.info("Place multiseed_violin.png in assets/")

    st.markdown("---")
    st.markdown("""
**Reporting template (copy into thesis/paper):**

> "All experiments were repeated across 5 independent random seeds (42, 123, 2024, 7, 99). 
> Hyperparameters were fixed to the Optuna best configuration from seed=42. 
> Results are reported as mean ± standard deviation across seeds."

**Seeds:** 42 · 123 · 2024 · 7 · 99  
**Dataset:** Kermany et al. (Cell 2018) · 84,495 OCT images · 4 classes  
**Test set:** 10,933 images  
    """)
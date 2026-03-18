"""
generate_demo.py
Run once locally on RTX 4060 to create demo_results.json for HuggingFace.

Usage:
    conda activate oct_dashboard
    python generate_demo.py
"""

import os
import json
import base64
import numpy as np
import cv2
import tensorflow as tf
import keras
import xgboost as xgb
import albumentations as A
from glob import glob
from scipy.spatial.distance import mahalanobis

# ── Custom Keras stubs ────────────────────────────────────────────────────────
class _DummyFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma, self.alpha = gamma, alpha
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
        self.seq_len, self.proj_dim = seq_len, proj_dim
    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding', shape=(1, self.seq_len, self.proj_dim),
            initializer='random_normal', trainable=True)
        super().build(input_shape)
    def call(self, x):
        return x + tf.cast(self.pos_embedding, dtype=x.dtype)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'seq_len': self.seq_len, 'proj_dim': self.proj_dim})
        return cfg


# ── Grad-CAM helpers ──────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer='top_activation'):
    """Guaranteed 2D float output."""
    try:
        conv_layer = model.get_layer(last_conv_layer)
    except ValueError:
        conv_layer = next(
            (l for l in model.layers
             if 'activation' in l.name and len(l.output_shape) == 4), None)
        if conv_layer is None:
            return np.zeros((7, 7), dtype=np.float32)

    grad_model = tf.keras.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output])

    # FIX: pass input as named dict to match model's expected input structure
    input_name = model.inputs[0].name.split(":")[0]
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(
            {input_name: img_array}, training=False)
        class_score = predictions[:, tf.argmax(predictions[0])]
    grads        = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = tf.nn.relu(tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis]))
    heatmap     /= (tf.reduce_max(heatmap) + 1e-8)
    res          = heatmap.numpy()

    # FIX: guarantee 2D before cv2.resize
    if res.ndim == 0:
        res = res.reshape(1, 1)
    elif res.ndim == 1:
        s   = max(1, int(np.sqrt(len(res))))
        res = res[:s*s].reshape(s, s)
    return res.astype(np.float32)


def overlay_gradcam(original_rgb, heatmap):
    """2D guard before cv2.resize."""
    heatmap = np.array(heatmap, dtype=np.float32)
    if heatmap.ndim == 0:
        heatmap = heatmap.reshape(1, 1)
    elif heatmap.ndim == 1:
        s       = max(1, int(np.sqrt(len(heatmap))))
        heatmap = heatmap[:s*s].reshape(s, s)
    h_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    h_col     = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    return np.uint8(0.4 * h_col + 0.6 * original_rgb)


def image_to_base64(img_rgb):
    _, buffer = cv2.imencode(
        '.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def apply_temperature(probs, T):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    scaled    = log_probs / T
    shifted   = scaled - scaled.max()
    exp_s     = np.exp(shifted)
    return exp_s / exp_s.sum()


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_demo_json():
    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading 2GB model into VRAM...")
    model = tf.keras.models.load_model(
        'models/Final_CNN_Transformer.keras',
        custom_objects={
            'FocalLoss'                   : _DummyFocalLoss,
            'LearnablePositionalEncoding' : LearnablePositionalEncoding,
        }
    )
    print(f"   Model loaded: {model.count_params():,} params")
    _iname = model.inputs[0].name
    input_name = _iname.split(":")[0]
    print(f"   Input name: {input_name}")

    feature_extractor = tf.keras.Model(
        inputs  = model.input,
        outputs = model.get_layer('feature_extraction_layer').output,
    )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/Final_XGBoost_Hybrid.json')

    temperature   = float(np.load('models/temperature.npy')[0])
    train_mean    = np.load('models/ood_train_mean.npy')
    cov_inv       = np.load('models/ood_cov_inv.npy')
    ood_threshold = float(np.load('models/ood_threshold.npy')[0])
    print(f"   Temperature T={temperature:.4f} | OOD threshold={ood_threshold:.2f}")

    # ── Preprocessing — must match training exactly ───────────────────────────
    val_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    demo_data   = {}

    # ── Process each sample scan ──────────────────────────────────────────────
    print("\nProcessing sample scans...")
    for cls in CLASS_NAMES:
        paths = (glob(f'sample_scans/{cls}/*.jpeg') +
                 glob(f'sample_scans/{cls}/*.jpg') +
                 glob(f'sample_scans/{cls}/*.png'))[:5]

        if not paths:
            print(f"   WARNING: No images in sample_scans/{cls}/ — skipping.")
            continue

        for path in paths:
            key = f"{cls}_{os.path.basename(path)}"
            print(f"   {key}...", end=' ', flush=True)

            try:
                raw_bgr = cv2.imread(path)
                if raw_bgr is None:
                    print("Could not read image — skipping.")
                    continue

                raw_rgb  = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                raw_r    = cv2.resize(raw_rgb, (224, 224))
                aug_img  = val_aug(image=raw_r)['image']
                x_in     = np.expand_dims(aug_img, axis=0).astype(np.float32)

                # Feature extraction
                feats = feature_extractor.predict(x_in, verbose=0)

                # OOD check
                dist   = mahalanobis(feats[0].flatten(), train_mean, cov_inv)
                is_ood = bool(dist >= ood_threshold)

                # XGBoost + temperature calibration
                raw_probs = xgb_model.predict_proba(feats)[0]
                cal_probs = apply_temperature(raw_probs, temperature)
                pred_idx  = int(np.argmax(cal_probs))
                confidence= float(cal_probs[pred_idx])
                probs_dict= {CLASS_NAMES[i]: float(cal_probs[i]) for i in range(4)}

                # MC Dropout — all 20 passes in ONE GPU call
                x_batch     = np.repeat(x_in, 20, axis=0)
                mc_preds    = model(x_batch, training=True).numpy()
                uncertainty = float(mc_preds.std(axis=0).max())

                # Grad-CAM
                heatmap = make_gradcam_heatmap(x_in, model)
                overlay = overlay_gradcam(raw_r, heatmap)

                demo_data[key] = {
                    'true_class'   : cls,
                    'prediction'   : CLASS_NAMES[pred_idx],
                    'confidence'   : confidence,
                    'probs'        : probs_dict,
                    'uncertainty'  : uncertainty,
                    'is_ood'       : is_ood,
                    'ood_score'    : float(dist),
                    'ood_threshold': ood_threshold,
                    'original_b64' : image_to_base64(raw_r),
                    'gradcam_b64'  : image_to_base64(overlay),
                }
                print(f"-> {CLASS_NAMES[pred_idx]} ({confidence:.1%}) | "
                      f"unc={uncertainty:.3f} | ood={is_ood}")

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback; traceback.print_exc()
                continue

    if not demo_data:
        print("\nERROR: No samples processed. Check sample_scans/ folder structure.")
        return

    with open('demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, indent=2)

    size_mb = os.path.getsize('demo_results.json') / 1e6
    print(f"\nDone: demo_results.json saved ({size_mb:.1f} MB, {len(demo_data)} samples)")
    print("Upload to HuggingFace Spaces alongside app.py.")
    print("Set DEMO_MODE=true in HuggingFace Space settings.")


if __name__ == "__main__":
    generate_demo_json()
import os
import tensorflow as tf
import tf2onnx

# 1. The Master Patch
if not hasattr(tf.keras.layers.Layer, "_is_patched"):
    original_layer_init = tf.keras.layers.Layer.__init__
    def patched_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) 
        original_layer_init(self, *args, **kwargs)
    tf.keras.layers.Layer.__init__ = patched_layer_init
    tf.keras.layers.Layer._is_patched = True

# 2. Custom Objects
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
        self.pos_embeddings = self.add_weight(shape=(1, seq_len, proj_dim), initializer="zeros", trainable=True, name="pos_embedding")
    def call(self, inputs):
        return inputs + self.pos_embeddings
    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "proj_dim": self.proj_dim})
        return config

# 3. Load Model and Convert
print("Loading 2GB Keras model (this takes a minute)...")
model = tf.keras.models.load_model(
    "Final_CNN_Transformer.keras",
    custom_objects={'FocalLoss': _DummyFocalLoss, 'LearnablePositionalEncoding': LearnablePositionalEncoding, 'Dense': tf.keras.layers.Dense},
    compile=False
)

print("Exporting to ONNX FP32... (Please wait)")
# Opset 13 handles standard gelu/math operations gracefully
input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input_image")]
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13, output_path="human_eye_fp32.onnx")

print("✅ Success! Your edge model (human_eye_fp32.onnx) is ready.")
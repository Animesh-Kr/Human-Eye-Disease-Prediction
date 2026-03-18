import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package='Custom')
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len=49, proj_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.seq_len, self.proj_dim = seq_len, proj_dim
        
    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding', 
            shape=(1, self.seq_len, self.proj_dim), 
            initializer='random_normal', 
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, x):
        return x + tf.cast(self.pos_embedding, dtype=x.dtype)
        
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'seq_len': self.seq_len, 'proj_dim': self.proj_dim})
        return cfg

def build_cnn_transformer(num_classes=4, img_size=224, n_heads=16, proj_dim=512, dropout=0.3, n_blocks=4):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name='input_scan')
    
    # Backbone
    backbone = tf.keras.applications.EfficientNetV2L(include_top=False, weights=None, input_tensor=inputs)
    cnn_features = backbone.output
    
    # Transformer Bridge
    x = tf.keras.layers.Reshape((49, 1280), name='patch_sequence')(cnn_features)
    x = tf.keras.layers.Dense(proj_dim, name='patch_projection')(x)
    x = LearnablePositionalEncoding(seq_len=49, proj_dim=proj_dim, name='positional_encoding')(x)
    
    # Attention Blocks
    for i in range(n_blocks):
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=proj_dim//n_heads)(x_norm, x_norm)
        x = tf.keras.layers.Add()([x, attn_out])
        
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff = tf.keras.layers.Dense(proj_dim * 4, activation='gelu')(x_norm)
        ff = tf.keras.layers.Dense(proj_dim)(ff)
        x = tf.keras.layers.Add()([x, ff])
        
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)
    pooled = tf.keras.layers.GlobalAveragePooling1D(name='feature_extraction_layer')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='disease_output', dtype='float32')(pooled)
    
    return tf.keras.Model(inputs=inputs, outputs=output)


import tensorflow as tf
import numpy as np
import cv2


def generate_gradcam(model, img_batch, intensity=0.5, res=224):
    """
    Generates a Grad-CAM heatmap for the EfficientNetV2L backbone.
    """
    # 1. Find the last conv layer of the backbone
    # EfficientNetV2L usually ends with 'top_activation' or 'post_relu'
    backbone = model.get_layer('efficientnetv2-l') 
    last_conv_layer = backbone.get_layer('top_activation')
    
    # 2. Create a model that maps input to (last_conv_output, final_predictions)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    # 3. Compute gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_batch)
        class_channel = preds[:, tf.argmax(preds[0])]

    # 4. Gradient of the predicted class wrt the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weigh the feature map by the gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 7. Resize and Colorize
    heatmap = cv2.resize(heatmap, (res, res))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 8. Overlay
    img = img_batch[0] * 255.0
    img = img.astype(np.uint8)
    overlayed_img = cv2.addWeighted(img, 1 - intensity, heatmap, intensity, 0)
    
    return overlayed_img
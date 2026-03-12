# ─────────────────────────────────────────────
#  model/gradcam.py
#  Grad-CAM implementation (Equations 14 & 15)
#  Highlights CT regions influencing prediction
# ─────────────────────────────────────────────

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMAGE_SIZE


def generate_gradcam(model, image: np.ndarray, class_idx: int = 1) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a CT scan image.

    Implements Equations 14 & 15 from paper:
      α_k^c = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}   (Eq. 15)
      L^c_Grad-CAM = ReLU( Σ_k α_k^c · A^k )     (Eq. 14)

    Args:
        model    : Trained Keras CNN model
        image    : Preprocessed image (224,224,1) float32
        class_idx: 1=Malignant, 0=Benign

    Returns:
        heatmap_overlay: RGB image with heatmap overlaid (224,224,3)
    """
    # ── Find last conv layer ──────────────────
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")

    # ── Build grad model ──────────────────────
    grad_model = tf.keras.models.Model(
        inputs  = model.input,
        outputs = [model.get_layer(last_conv).output, model.output]
    )

    # ── Compute gradients ─────────────────────
    img_batch = np.expand_dims(image, axis=0)   # (1,224,224,1)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, class_idx]         # score for target class

    # Gradient of class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)    # (1, H, W, K)

    # ── Equation 15: α_k^c ────────────────────
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (K,)

    # ── Equation 14: weighted sum ─────────────
    conv_outputs = conv_outputs[0]               # (H, W, K)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)                # (H, W)

    # Apply ReLU — only positive contributions
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # ── Upscale to original image size ────────
    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)

    # ── Create colored overlay ────────────────
    heatmap_uint8  = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original grayscale image to RGB
    img_display = (image[:, :, 0] * 255).astype(np.uint8)
    img_rgb     = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

    # Blend original image + heatmap
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap_resized


def save_gradcam(overlay: np.ndarray, save_path: str):
    """Save Grad-CAM overlay image to disk."""
    cv2.imwrite(save_path, overlay)
    print(f"[INFO] Grad-CAM saved → {save_path}")

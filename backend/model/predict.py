# ─────────────────────────────────────────────
#  model/predict.py
#  Standalone inference on a single CT image
#  Use this to test model before running Flask
# ─────────────────────────────────────────────

import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_SAVE_PATH, CLASS_NAMES
from preprocessing.preprocess import preprocess_image
from model.gradcam import generate_gradcam, save_gradcam
from model.risk_score import compute_risk_score


def predict_single(
    image_path: str,
    age: int            = 50,
    smoking_status: str = "never",
    symptoms: list      = None
) -> dict:
    """
    Run full inference pipeline on a single CT scan.

    Returns complete result dict with classification,
    risk score, and Grad-CAM path.
    """
    if symptoms is None:
        symptoms = []

    # ── Load model ────────────────────────────
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    # ── Preprocess ────────────────────────────
    print("[INFO] Preprocessing image...")
    processed = preprocess_image(image_path)

    # ── Predict ───────────────────────────────
    batch = np.expand_dims(processed, axis=0)
    preds = model.predict(batch, verbose=0)[0]

    benign_prob    = float(preds[0])
    malignant_prob = float(preds[1])
    class_idx      = int(np.argmax(preds))
    classification = CLASS_NAMES[class_idx]

    # ── Risk Score ────────────────────────────
    risk = compute_risk_score(malignant_prob, age, smoking_status, symptoms)

    # ── Grad-CAM ──────────────────────────────
    overlay, _ = generate_gradcam(model, processed, class_idx)
    gradcam_path = str(Path(image_path).parent / "gradcam_result.png")
    save_gradcam(overlay, gradcam_path)

    # ── Print Results ─────────────────────────
    print("\n" + "=" * 45)
    print("  PREDICTION RESULT")
    print("=" * 45)
    print(f"  Classification  : {classification}")
    print(f"  Benign Prob     : {benign_prob:.4f}")
    print(f"  Malignant Prob  : {malignant_prob:.4f}")
    print(f"  Risk Score      : {risk['risk_score']}")
    print(f"  Risk Tier       : {risk['risk_tier']}")
    print(f"  Grad-CAM saved  : {gradcam_path}")
    print("=" * 45)

    return {
        "classification":  classification,
        "benign_prob":     benign_prob,
        "malignant_prob":  malignant_prob,
        "risk":            risk,
        "gradcam_path":    gradcam_path
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict_single(sys.argv[1], age=60, smoking_status="current",
                       symptoms=["persistent_cough"])
    else:
        print("Usage: python predict.py <path_to_ct_image>")

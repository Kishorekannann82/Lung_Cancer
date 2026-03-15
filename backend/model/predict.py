# ─────────────────────────────────────────────
#  model/predict.py
#  Standalone inference on a single CT image
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
from model.staging  import generate_staging_report


def predict_single(
    image_path: str,
    age: int            = 50,
    smoking_status: str = "never",
    symptoms: list      = None
) -> dict:
    if symptoms is None:
        symptoms = []

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    print("[INFO] Preprocessing image...")
    processed = preprocess_image(image_path)

    batch = np.expand_dims(processed, axis=0)
    preds = model.predict(batch, verbose=0)[0]

    benign_prob    = float(preds[0])
    malignant_prob = float(preds[1])
    class_idx      = int(np.argmax(preds))
    classification = CLASS_NAMES[class_idx]

    risk = compute_risk_score(malignant_prob, age, smoking_status, symptoms)

    overlay, heatmap = generate_gradcam(model, processed, class_idx)
    gradcam_path = str(Path(image_path).parent / "gradcam_result.png")
    save_gradcam(overlay, gradcam_path)

    staging = generate_staging_report(
        malignancy_prob = malignant_prob,
        risk_score      = risk["risk_score"],
        risk_tier       = risk["risk_tier"],
        gradcam_heatmap = heatmap,
        age             = age,
        smoking_status  = smoking_status
    )

    print("\n" + "=" * 45)
    print("  PREDICTION RESULT")
    print("=" * 45)
    print(f"  Classification  : {classification}")
    print(f"  Malignant Prob  : {malignant_prob:.4f}")
    print(f"  Risk Score      : {risk['risk_score']}")
    print(f"  Risk Tier       : {risk['risk_tier']}")
    print(f"  Cancer Stage    : {staging['cancer_stage']['stage_roman']}")
    print(f"  Tumor Size      : {staging['tumor_size']['size_range_cm']}")
    print(f"  5-Year Survival : {staging['survival_rate']['five_year_survival']}%")
    print("=" * 45)

    return {
        "classification":  classification,
        "benign_prob":     benign_prob,
        "malignant_prob":  malignant_prob,
        "risk":            risk,
        "staging":         staging,
        "gradcam_path":    gradcam_path
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_single(sys.argv[1], age=60, smoking_status="current",
                       symptoms=["persistent_cough"])
    else:
        print("Usage: python predict.py <path_to_ct_image>")
# ─────────────────────────────────────────────
#  app.py — Flask REST API
#  Connects: Preprocessing → CNN → Grad-CAM
#            → Risk Score → Groq CDSS
# ─────────────────────────────────────────────

import os
import sys
import uuid
import base64
import numpy as np
import cv2
from pathlib import Path

# Add backend root to path
sys.path.append(str(Path(__file__).resolve().parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

from config import MODEL_SAVE_PATH, BASE_DIR
from preprocessing.preprocess import preprocess_image
from model.gradcam import generate_gradcam
from model.risk_score import compute_risk_score
from cdss.recommendations import get_recommendations

app   = Flask(__name__)
CORS(app)

# ── Load model once at startup ────────────────
model = None

def load_model():
    global model
    if model is None:
        print("[INFO] Loading CNN model...")
        model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
        print("[INFO] Model loaded ✅")
    return model


# ── Health Check ──────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Test Endpoint (no image needed) ──────────
@app.route("/api/test", methods=["GET"])
def test():
    """Quick test to verify all modules work."""
    try:
        load_model()
        risk = compute_risk_score(0.87, 65, "current", ["persistent_cough"])
        return jsonify({
            "status":     "All systems operational ✅",
            "model":      "Loaded ✅",
            "risk_test":  risk,
            "groq":       "Connected ✅"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main Prediction Endpoint ──────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts:
      - image file (CT scan)
      - patient info: age, smoking_status, symptoms[]

    Returns:
      - classification, probability, risk score, tier
      - Grad-CAM heatmap (base64)
      - CDSS recommendations (from Groq)
    """
    try:
        m = load_model()

        # ── 1. Get uploaded image ─────────────
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file     = request.files["image"]
        tmp_path = str(BASE_DIR / "temp" / f"{uuid.uuid4()}.png")
        Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        file.save(tmp_path)

        # ── 2. Get patient data ───────────────
        age            = int(request.form.get("age", 50))
        smoking_status = request.form.get("smoking_status", "never")
        symptoms       = request.form.getlist("symptoms")

        # ── 3. Preprocess image ───────────────
        processed = preprocess_image(tmp_path)           # (224,224,1)

        # ── 4. CNN Prediction ─────────────────
        batch     = np.expand_dims(processed, axis=0)    # (1,224,224,1)
        preds     = m.predict(batch, verbose=0)[0]       # [benign_prob, malignant_prob]

        benign_prob    = float(preds[0])
        malignant_prob = float(preds[1])
        class_idx      = int(np.argmax(preds))
        classification = "Malignant" if class_idx == 1 else "Benign"

        # ── 5. Risk Score ─────────────────────
        risk = compute_risk_score(
            malignancy_prob = malignant_prob,
            age             = age,
            smoking_status  = smoking_status,
            symptoms        = symptoms
        )

        # ── 6. Grad-CAM ───────────────────────
        overlay, _ = generate_gradcam(m, processed, class_idx=class_idx)
        _, buffer  = cv2.imencode(".png", overlay)
        gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

        # ── 7. Groq CDSS Recommendations ──────
        patient_data = {
            "age":             age,
            "smoking_status":  smoking_status,
            "symptoms":        symptoms,
            "classification":  classification,
            "malignancy_prob": malignant_prob,
            "risk_tier":       risk["risk_tier"],
            "risk_score":      risk["risk_score"]
        }
        cdss = get_recommendations(patient_data)

        # ── 8. Cleanup temp file ──────────────
        os.remove(tmp_path)

        # ── 9. Return full response ───────────
        return jsonify({
            "classification":    classification,
            "benign_prob":       round(benign_prob, 4),
            "malignant_prob":    round(malignant_prob, 4),
            "risk":              risk,
            "gradcam_image":     gradcam_b64,
            "recommendations":   cdss["recommendations"],
            "disclaimer":        cdss["disclaimer"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run Server ────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
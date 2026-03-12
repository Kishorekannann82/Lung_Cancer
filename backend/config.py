# ─────────────────────────────────────────────
#  config.py — Central config for all modules
# ─────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────
BASE_DIR        = Path("D:/Lung_Cancer")
DATASET_DIR     = BASE_DIR / "The IQ-OTHNCCD lung cancer dataset"

RAW_MALIGNANT   = DATASET_DIR / "Malignant cases"
RAW_BENIGN      = DATASET_DIR / "Bengin cases"    # note: typo in dataset ("Bengin")
RAW_NORMAL      = DATASET_DIR / "Normal cases"    # treated as Benign

PROCESSED_DIR   = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = BASE_DIR / "model" / "cnn_model.h5"
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Image Settings ────────────────────────────
IMAGE_SIZE      = (224, 224)
CHANNELS        = 1                # Grayscale CT scans
INPUT_SHAPE     = (224, 224, 1)

# ── Dataset Split ─────────────────────────────
TRAIN_RATIO     = 0.8
TEST_RATIO      = 0.2
RANDOM_SEED     = 42

# ── Training Hyperparameters ──────────────────
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 0.001

# ── Risk Score Weights (Equation 12 from paper)
# R = α1*P(M|x) + α2*C_age + α3*C_smoke + α4*C_hist
ALPHA_1         = 0.5
ALPHA_2         = 0.2
ALPHA_3         = 0.2
ALPHA_4         = 0.1

# ── Risk Thresholds (Equation 13 from paper) ──
RISK_LOW        = 0.35
RISK_MEDIUM     = 0.55
RISK_HIGH       = 0.75

# ── Groq API ──────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama3-70b-8192"

# ── Labels ────────────────────────────────────
LABEL_MAP       = {1: "Malignant", 0: "Benign"}
CLASS_NAMES     = ["Benign", "Malignant"]

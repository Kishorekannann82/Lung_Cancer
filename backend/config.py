# ─────────────────────────────────────────────
#  config.py — Central config for all modules
# ─────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from backend directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Fallback: manually parse .env if dotenv fails
if not os.getenv("GROQ_API_KEY"):
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "GROQ_API_KEY" in line and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["GROQ_API_KEY"] = key
                    break
    except Exception:
        pass

# ── Paths ─────────────────────────────────────
# Works both locally (D:\Lung_Cancer) and on Railway
import sys

if sys.platform == "win32":
    BASE_DIR = Path("D:/Lung_Cancer")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent  # Railway Linux path

DATASET_DIR     = BASE_DIR / "The IQ-OTHNCCD lung cancer dataset"
RAW_MALIGNANT   = DATASET_DIR / "Malignant cases"
RAW_BENIGN      = DATASET_DIR / "Bengin cases"
RAW_NORMAL      = DATASET_DIR / "Normal cases"

PROCESSED_DIR   = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = Path(__file__).resolve().parent.parent / "model" / "cnn_model.h5"
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
BATCH_SIZE      = 16     # Smaller batch = less RAM on CPU
EPOCHS          = 25     # EarlyStopping will halt before this if needed
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
GROQ_MODEL      = "llama-3.3-70b-versatile"

# ── Labels ────────────────────────────────────
LABEL_MAP       = {1: "Malignant", 0: "Benign"}
CLASS_NAMES     = ["Benign", "Malignant"]
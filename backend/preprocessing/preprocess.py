# ─────────────────────────────────────────────
#  preprocessing/preprocess.py
#  Pipeline: Denoise → Resize → Normalize
#  Implements Equations 1, 2, 3 from paper
# ─────────────────────────────────────────────

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMAGE_SIZE, PROCESSED_DIR, RAW_MALIGNANT, RAW_BENIGN, RAW_NORMAL


# ── Step 1: Denoise — Equation 2 ─────────────
def denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing filter.
    I_denoised(x,y) = integral[ I_raw * G_sigma ] dx'dy'
    """
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), sigma)


# ── Step 2: Resize ────────────────────────────
def resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)


# ── Step 3: Normalize — Equation 3 ───────────
def normalize(image: np.ndarray) -> np.ndarray:
    """
    I_norm = (I - I_min) / (I_max - I_min)
    Scales pixel values to [0, 1]
    """
    i_min, i_max = image.min(), image.max()
    if i_max - i_min == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image.astype(np.float32) - i_min) / (i_max - i_min)


# ── Step 4: CLAHE Contrast Enhancement ───────
def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Improve nodule visibility using CLAHE."""
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_uint8)
    return enhanced.astype(np.float32) / 255.0


# ── Full Pipeline — Equation 1 ────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """
    I_processed = Normalize( Resize( Denoise( I_raw ) ) )
    Returns shape (224, 224, 1) float32
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Cannot load image: {image_path}")

    processed = denoise(raw)
    processed = resize(processed)
    processed = normalize(processed)
    processed = enhance_contrast(processed)

    return np.expand_dims(processed, axis=-1)   # (224, 224, 1)


# ── Batch Process a Folder ────────────────────
def process_folder(src_folder: Path, label: int, out_dir: Path) -> list:
    """Process all images in a folder and save as .npy files."""
    label_name = "malignant" if label == 1 else "benign"
    save_dir   = out_dir / label_name
    save_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    files      = [f for f in src_folder.iterdir() if f.suffix.lower() in extensions]

    results = []
    print(f"\n[INFO] Processing '{label_name}' — {len(files)} images from {src_folder}")

    for f in tqdm(files, desc=f"  {label_name}"):
        try:
            img       = preprocess_image(f)
            save_path = save_dir / (f.stem + ".npy")
            np.save(str(save_path), img)
            results.append((img, label))
        except Exception as e:
            print(f"  [WARN] Skipping {f.name}: {e}")

    return results


# ── Load Preprocessed Dataset ─────────────────
def load_dataset() -> tuple:
    """
    Load all .npy files from processed dir.
    Returns X (N,224,224,1) and y (N,) arrays.
    """
    X, y = [], []

    for label_name, label_val in [("malignant", 1), ("benign", 0)]:
        folder = PROCESSED_DIR / label_name
        if not folder.exists():
            print(f"[WARN] {folder} not found — run preprocess first.")
            continue
        files = list(folder.glob("*.npy"))
        for f in tqdm(files, desc=f"  Loading {label_name}"):
            X.append(np.load(str(f)))
            y.append(label_val)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n[INFO] Dataset loaded:")
    print(f"  Total     : {len(y)}")
    print(f"  Malignant : {y.sum()}")
    print(f"  Benign    : {(y == 0).sum()}")
    return X, y


# ── CLI Runner ────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  PHASE 1 — Preprocessing Pipeline")
    print("=" * 50)

    # Process Malignant
    process_folder(RAW_MALIGNANT, label=1, out_dir=PROCESSED_DIR)

    # Process Benign
    process_folder(RAW_BENIGN, label=0, out_dir=PROCESSED_DIR)

    # Process Normal → treat as Benign
    process_folder(RAW_NORMAL, label=0, out_dir=PROCESSED_DIR)

    print("\n[✓] All images preprocessed and saved!")
    print(f"    Output → {PROCESSED_DIR}")

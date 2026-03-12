# ─────────────────────────────────────────────
#  preprocessing/dataset_split.py
#  Loads preprocessed .npy files and saves
#  80/20 train/test splits
# ─────────────────────────────────────────────

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, BASE_DIR, TRAIN_RATIO, RANDOM_SEED
from preprocess import load_dataset


def split_and_save():
    print("=" * 50)
    print("  Splitting Dataset — 80% Train / 20% Test")
    print("=" * 50)

    # Load all preprocessed images
    X, y = load_dataset()

    if len(X) == 0:
        print("[ERROR] No data found! Run preprocess.py first.")
        return

    # Train / Test split (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = 1 - TRAIN_RATIO,
        random_state= RANDOM_SEED,
        stratify    = y
    )

    # Save splits
    out = BASE_DIR / "splits"
    out.mkdir(parents=True, exist_ok=True)

    np.save(str(out / "X_train.npy"), X_train)
    np.save(str(out / "X_test.npy"),  X_test)
    np.save(str(out / "y_train.npy"), y_train)
    np.save(str(out / "y_test.npy"),  y_test)

    print(f"\n[✓] Splits saved to {out}")
    print(f"    X_train : {X_train.shape}  y_train: {y_train.shape}")
    print(f"    X_test  : {X_test.shape}   y_test : {y_test.shape}")
    print(f"\n    Train — Malignant: {y_train.sum()}  Benign: {(y_train==0).sum()}")
    print(f"    Test  — Malignant: {y_test.sum()}   Benign: {(y_test==0).sum()}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    split_and_save()

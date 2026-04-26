# ─────────────────────────────────────────────
#  model/save_augmented.py
#  Generates and saves augmented CT scan images
#  to disk as PNG files for inspection & reuse.
#
#  Run once before training:
#  python model/save_augmented.py
# ─────────────────────────────────────────────

import numpy as np
import cv2
import os
import sys
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR

SPLITS_DIR       = BASE_DIR / "splits"
AUG_DIR          = BASE_DIR / "augmented"
TARGET_PER_CLASS = 2000   # change to 1500 if low RAM


def build_augmentor():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )


def save_augmented_images():
    print("\n" + "="*50)
    print("  Augmented Image Generator — LungAI")
    print("="*50 + "\n")

    print("[INFO] Loading splits...")
    X_train = np.load(str(SPLITS_DIR / "X_train.npy"))
    y_train = np.load(str(SPLITS_DIR / "y_train.npy"))
    print(f"[INFO] Loaded {len(X_train)} training samples")

    class_names = {0: "benign", 1: "malignant"}
    for cls_name in class_names.values():
        (AUG_DIR / "original"  / cls_name).mkdir(parents=True, exist_ok=True)
        (AUG_DIR / "augmented" / cls_name).mkdir(parents=True, exist_ok=True)

    augmentor = build_augmentor()

    for cls_idx, cls_name in class_names.items():
        cls_mask = y_train == cls_idx
        X_cls    = X_train[cls_mask]
        current  = len(X_cls)
        needed   = TARGET_PER_CLASS - current

        print(f"\n[INFO] Class: {cls_name.upper()}")
        print(f"       Original : {current} | Target : {TARGET_PER_CLASS} | Generate : {needed}")

        # Save originals
        orig_dir = AUG_DIR / "original" / cls_name
        for i, img in enumerate(X_cls):
            img_uint8 = (img * 255).astype(np.uint8).squeeze()
            cv2.imwrite(str(orig_dir / f"{cls_name}_orig_{i:04d}.png"), img_uint8)
        print(f"       Saved {current} originals → augmented/original/{cls_name}/")

        # Save augmented
        aug_dir = AUG_DIR / "augmented" / cls_name
        gen     = augmentor.flow(X_cls, batch_size=1, shuffle=True, seed=42)
        saved   = 0
        while saved < needed:
            batch     = next(gen)
            img_uint8 = (np.clip(batch[0], 0, 1) * 255).astype(np.uint8).squeeze()
            cv2.imwrite(str(aug_dir / f"{cls_name}_aug_{saved:04d}.png"), img_uint8)
            saved += 1
            if saved % 200 == 0:
                print(f"       ... {saved}/{needed} saved")
        print(f"       Saved {needed} augmented → augmented/augmented/{cls_name}/")

    # Save combined .npy
    print("\n[INFO] Saving combined .npy for fast reloading...")
    X_all, y_all = [], []
    for cls_idx, cls_name in class_names.items():
        for folder in ["original", "augmented"]:
            for f in sorted((AUG_DIR / folder / cls_name).glob("*.png")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                X_all.append(img[..., np.newaxis])
                y_all.append(cls_idx)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    idx   = np.random.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    np.save(str(AUG_DIR / "X_train_augmented.npy"), X_all)
    np.save(str(AUG_DIR / "y_train_augmented.npy"), y_all)

    print(f"\n{'='*50}")
    print(f"  Original  : {len(X_train)} samples")
    print(f"  Augmented : {len(X_all)} samples")
    print(f"  Malignant : {(y_all==1).sum()} | Benign: {(y_all==0).sum()}")
    print(f"\n  Saved to: {AUG_DIR}")
    print(f"  augmented/")
    print(f"  ├── original/malignant/   ({(y_train==1).sum()} PNGs)")
    print(f"  ├── original/benign/      ({(y_train==0).sum()} PNGs)")
    print(f"  ├── augmented/malignant/  ({TARGET_PER_CLASS-(y_train==1).sum()} PNGs)")
    print(f"  ├── augmented/benign/     ({TARGET_PER_CLASS-(y_train==0).sum()} PNGs)")
    print(f"  ├── X_train_augmented.npy")
    print(f"  └── y_train_augmented.npy")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    save_augmented_images()
# ─────────────────────────────────────────────
#  model/train.py
#  Trains the CNN on preprocessed CT scans
#  and saves the model weights
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, MODEL_SAVE_PATH, BATCH_SIZE, EPOCHS
from model.cnn_model import build_cnn


def load_splits():
    """Load the 80/20 train/test splits."""
    splits_dir = BASE_DIR / "splits"
    X_train = np.load(str(splits_dir / "X_train.npy"))
    X_test  = np.load(str(splits_dir / "X_test.npy"))
    y_train = np.load(str(splits_dir / "y_train.npy"))
    y_test  = np.load(str(splits_dir / "y_test.npy"))
    return X_train, X_test, y_train, y_test


def plot_history(history, save_dir: Path):
    """Save training accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(str(save_dir / "training_curves.png"))
    print(f"[INFO] Training curves saved → {save_dir / 'training_curves.png'}")
    plt.close()


def train():
    print("=" * 50)
    print("  PHASE 2 — CNN Model Training")
    print("=" * 50)

    # ── Load Data ─────────────────────────────
    print("\n[INFO] Loading dataset splits...")
    X_train, X_test, y_train, y_test = load_splits()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Build Model ───────────────────────────
    print("\n[INFO] Building CNN...")
    model = build_cnn()
    model.summary()

    # ── Callbacks ─────────────────────────────
    callbacks = [
        # Stop early if val_loss stops improving
        EarlyStopping(
            monitor='val_loss', patience=7,
            restore_best_weights=True, verbose=1
        ),
        # Save best model automatically
        ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        # Reduce LR when plateau
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        )
    ]

    # ── Train ─────────────────────────────────
    print("\n[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 1
    )

    # ── Evaluate ──────────────────────────────
    print("\n[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")

    # ── Save curves ───────────────────────────
    plot_history(history, MODEL_SAVE_PATH.parent)

    print(f"\n[✓] Model saved → {MODEL_SAVE_PATH}")
    print("→ Ready for Phase 3: Grad-CAM + Risk Score!")

    return model, history


if __name__ == "__main__":
    train()

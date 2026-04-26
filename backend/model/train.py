# ─────────────────────────────────────────────
#  model/train.py
#  CNN Training Pipeline with Data Augmentation
#  Dataset: IQ-OTH/NCCD (1,097 CT scans)
#  Augmented to ~5,000+ samples before training
# ─────────────────────────────────────────────

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_SAVE_PATH, BATCH_SIZE, EPOCHS,
    BASE_DIR
)

SPLITS_DIR = BASE_DIR / "splits"


# ── Load Preprocessed Splits ──────────────────
def load_splits():
    print("[INFO] Loading preprocessed splits...")
    X_train = np.load(str(SPLITS_DIR / "X_train.npy"))
    X_test  = np.load(str(SPLITS_DIR / "X_test.npy"))
    y_train = np.load(str(SPLITS_DIR / "y_train.npy"))
    y_test  = np.load(str(SPLITS_DIR / "y_test.npy"))

    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Malignant train: {y_train.sum()} | Benign train: {len(y_train)-y_train.sum()}")
    return X_train, X_test, y_train, y_test


# ── Data Augmentation ─────────────────────────
def build_augmentor():
    """
    Augmentation pipeline for CT scan images.
    Conservative settings — preserves medical image integrity.
    """
    return ImageDataGenerator(
        rotation_range=15,          # slight rotation (lung orientation)
        width_shift_range=0.08,     # small horizontal shift
        height_shift_range=0.08,    # small vertical shift
        zoom_range=0.08,            # slight zoom in/out
        horizontal_flip=True,       # valid for CT axial slices
        brightness_range=[0.9, 1.1],# minor intensity variation
        fill_mode='nearest'         # fill empty pixels with nearest value
    )


def augment_dataset(X_train, y_train, target_per_class=2000):
    """
    Augment training data to target_per_class samples per class.
    Default: 2000 per class → ~4000 total training samples.
    """
    augmentor = build_augmentor()

    X_aug_list = [X_train]
    y_aug_list = [y_train]

    classes = [0, 1]  # 0=Benign, 1=Malignant
    class_names = {0: "Benign", 1: "Malignant"}

    for cls in classes:
        cls_mask   = y_train == cls
        X_cls      = X_train[cls_mask]
        current_n  = len(X_cls)
        needed     = target_per_class - current_n

        if needed <= 0:
            print(f"[INFO] {class_names[cls]}: {current_n} samples — no augmentation needed")
            continue

        print(f"[INFO] {class_names[cls]}: {current_n} → generating {needed} augmented samples...")

        X_new = []
        y_new = []
        gen   = augmentor.flow(X_cls, batch_size=1, shuffle=True)

        while len(X_new) < needed:
            batch = next(gen)
            X_new.append(batch[0])
            y_new.append(cls)

        X_aug_list.append(np.array(X_new))
        y_aug_list.append(np.array(y_new))

    X_augmented = np.concatenate(X_aug_list, axis=0)
    y_augmented = np.concatenate(y_aug_list, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[idx]
    y_augmented = y_augmented[idx]

    print(f"\n[INFO] ✅ Augmentation complete!")
    print(f"[INFO] Original:  {len(X_train)} samples")
    print(f"[INFO] Augmented: {len(X_augmented)} samples")
    print(f"[INFO] Malignant: {y_augmented.sum()} | Benign: {len(y_augmented)-y_augmented.sum()}")

    return X_augmented, y_augmented


# ── Build CNN Model ───────────────────────────
def build_model(input_shape=(224, 224, 1)):
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # Block 2
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # Block 3
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # Head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ], name="LungAI_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Callbacks ─────────────────────────────────
def build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_SAVE_PATH), monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        )
    ]


# ── Plot Training Curves ──────────────────────
def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True)

    save_path = str(MODEL_SAVE_PATH.parent / "training_curves.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


# ── Evaluate Model ────────────────────────────
def evaluate(model, X_test, y_test):
    print("\n[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*45}")
    print(f"  TEST ACCURACY : {acc*100:.2f}%")
    print(f"  TEST LOSS     : {loss:.4f}")
    print(f"{'='*45}\n")

    # Per-class report
    from sklearn.metrics import classification_report
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, preds, target_names=["Benign", "Malignant"]))


# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*45)
    print("  LungAI CNN Training with Augmentation")
    print("="*45 + "\n")

    # 1. Load data
    X_train, X_test, y_train, y_test = load_splits()

    # 2. Augment training data
    #    target_per_class=2000 → ~4000 total training samples
    #    Change to 1500 if RAM is limited
    X_train_aug, y_train_aug = augment_dataset(
        X_train, y_train,
        target_per_class=2000
    )

    # 3. Build model
    model = build_model(input_shape=(224, 224, 1))
    model.summary()

    # 4. Train
    print(f"\n[INFO] Training on {len(X_train_aug)} samples...")
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(),
        verbose=1
    )

    # 5. Evaluate
    evaluate(model, X_test, y_test)

    # 6. Plot curves
    plot_curves(history)

    print(f"[INFO] Model saved → {MODEL_SAVE_PATH}")
    print("[INFO] Training complete ✅")
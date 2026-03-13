# ─────────────────────────────────────────────
#  model/cnn_model.py
#  CNN Architecture (Fig. 3 from paper)
#  Conv → Pool → Conv → Pool → Conv → GAP
#  → Dense(256) → Dense(128) → Softmax(2)
# ─────────────────────────────────────────────

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import INPUT_SHAPE, LEARNING_RATE


def build_cnn() -> Sequential:
    """
    CNN architecture matching Fig. 3 from the paper.

    Layer stack:
      Input (224,224,1)
      → Conv2D(32, 3x3) + BN + ReLU + MaxPool(2x2)
      → Conv2D(64, 3x3) + BN + ReLU + MaxPool(2x2)
      → Conv2D(128,3x3) + BN + ReLU + MaxPool(2x2)
      → GlobalAveragePooling
      → Dense(256, ReLU) + Dropout(0.5)
      → Dense(128, ReLU) + Dropout(0.3)
      → Dense(2, Softmax)   ← Benign / Malignant
    """
    model = Sequential([

        # ── Block 1 ──────────────────────────
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=INPUT_SHAPE, name='conv1'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='pool1'),

        # ── Block 2 ──────────────────────────
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='pool2'),

        # ── Block 3 ──────────────────────────
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='pool3'),

        # ── Global Average Pooling ────────────
        GlobalAveragePooling2D(name='gap'),

        # ── Fully Connected Layers ────────────
        Dense(128, activation='relu', name='fc1'),
        Dropout(0.5),

        Dense(64, activation='relu', name='fc2'),
        Dropout(0.3),

        # ── Output — Softmax (2 classes) ──────
        Dense(2, activation='softmax', name='output')

    ], name='LungCancer_CNN')

    # ── Compile with Adam optimizer ───────────
    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()

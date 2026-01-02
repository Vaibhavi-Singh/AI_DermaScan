
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 20
NUM_CLASSES = 4

DATA_DIR = Path("data/processed_aug")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "efficientnet_b0.keras"

# =========================
# DATA GENERATORS
# =========================
def make_generators(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir / "train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        data_dir / "val",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        data_dir / "test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# =========================
# CLASS WEIGHTS
# =========================
def get_class_weights(train_gen):
    class_ids = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_ids),
        y=class_ids
    )
    return dict(enumerate(class_weights))

# =========================
# MODEL BUILDING
# =========================
def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3)
    )

    base_model.trainable = False  # PHASE 1

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model

# =========================
# TRAINING PLOTS
# =========================
def save_training_plot(history1, history2, save_path):
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📊 Training plot saved at: {save_path}")

# =========================
# TRAINING
# =========================
def train():
    train_gen, val_gen, test_gen = make_generators(DATA_DIR)
    class_weights = get_class_weights(train_gen)

    model, base_model = build_model()

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
       ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n🚀 PHASE 1: Training top layers only\n")

    history_phase_1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE_1,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print("\n🔥 PHASE 2: Fine-tuning last 30 EfficientNet layers\n")

    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase_2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE_2,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # =========================
    # SAVE TRAINING PLOTS
    # =========================
    PLOT_PATH = MODEL_DIR / "training_plot.png"
    save_training_plot(history_phase_1, history_phase_2, PLOT_PATH)

    print("\n✅ Training completed successfully!")
    model.save(MODEL_PATH)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()
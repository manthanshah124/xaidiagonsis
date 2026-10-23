#!/usr/bin/env python3
"""
Quick training on tiny dataset (3 images/class).
Usage: python ml/train_tiny.py --epochs 10
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
from sklearn.utils.class_weight import compute_class_weight

# Tiny dataset config
TINY_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed_tiny"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

SELECTED_CLASSES = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus", "Apple_Apple_scab", "Apple_Black_rot",
    "Apple_Cedar_apple_rust", "Apple_healthy",
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4  # Tiny dataset = tiny batches
NUM_CLASSES = len(SELECTED_CLASSES)


def create_datasets(data_dir: Path, batch_size: int = BATCH_SIZE):
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 127.5 - 1.0, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'train',
        labels='inferred', label_mode='int', class_names=SELECTED_CLASSES,
        color_mode='rgb', batch_size=batch_size, image_size=IMAGE_SIZE,
        shuffle=True, seed=42, interpolation='bilinear'
    ).map(normalize).map(augment).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'val',
        labels='inferred', label_mode='int', class_names=SELECTED_CLASSES,
        color_mode='rgb', batch_size=batch_size, image_size=IMAGE_SIZE,
        shuffle=False, seed=42, interpolation='bilinear'
    ).map(normalize).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'test',
        labels='inferred', label_mode='int', class_names=SELECTED_CLASSES,
        color_mode='rgb', batch_size=batch_size, image_size=IMAGE_SIZE,
        shuffle=False, seed=42, interpolation='bilinear'
    ).map(normalize).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model():
    base_model = applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet', pooling=None
    )
    base_model.trainable = False  # Stage 1: frozen

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_tiny")
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    return model


def get_callbacks(model_path: Path, patience: int = 5):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(model_path), monitor='val_loss', save_best_only=True, verbose=1),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Total epochs (Stage 1 + Stage 2)')
    parser.add_argument('--stage1-epochs', type=int, default=5, help='Frozen epochs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fine-tune-layers', type=int, default=30)
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load class names
    with open(MODELS_DIR / "class_names.json", 'r') as f:
        class_names = json.load(f)
    print(f"Classes: {class_names}")

    # Create datasets
    print("Creating datasets...")
    train_ds, val_ds, test_ds = create_datasets(TINY_DATA_DIR, args.batch_size)

    # Class weights (balanced, but tiny dataset = all weights ~1)
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(y.numpy())
    labels = np.array(labels)
    class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=labels)
    class_weights = {i: float(w) for i, w in enumerate(class_weights)}
    print("Class weights:", {class_names[k]: v for k, v in class_weights.items()})

    # Build model
    model, base_model = build_model()

    # Stage 1: Frozen
    print(f"\n{'='*50}")
    print(f"STAGE 1: Frozen backbone ({args.stage1_epochs} epochs)")
    print(f"{'='*50}")
    model = compile_model(model, args.lr)
    model.fit(train_ds, validation_data=val_ds, epochs=args.stage1_epochs,
              class_weight=class_weights, callbacks=get_callbacks(MODELS_DIR / "plant_model_tiny.keras"),
              verbose=1)

    # Stage 2: Fine-tune
    stage2_epochs = args.epochs - args.stage1_epochs
    if stage2_epochs > 0:
        print(f"\n{'='*50}")
        print(f"STAGE 2: Fine-tune top {args.fine_tune_layers} layers ({stage2_epochs} epochs)")
        print(f"{'='*50}")

        fine_tune_from = len(base_model.layers) - args.fine_tune_layers
        for layer in base_model.layers[fine_tune_from:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        for layer in base_model.layers[fine_tune_from:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model = compile_model(model, args.lr * 0.01)  # 10x lower LR
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"Trainable params: {trainable_params}")

        model.fit(train_ds, validation_data=val_ds, epochs=stage2_epochs,
                  class_weight=class_weights, callbacks=get_callbacks(MODELS_DIR / "plant_model_tiny.keras"),
                  verbose=1)

    # Save final model
    final_path = MODELS_DIR / "plant_model_tiny.keras"
    model.save(final_path)
    print(f"\nModel saved to {final_path}")

    # Save gradcam layer
    with open(MODELS_DIR / "gradcam_layer.txt", 'w') as f:
        f.write('Conv_1')

    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, Top-3: {results[2]:.4f}")

    # Quick prediction test
    print("\nSample predictions:")
    for images, labels in test_ds.take(1):
        preds = model.predict(images, verbose=0)
        for i in range(min(5, len(images))):
            true_cls = class_names[labels[i].numpy()]
            pred_cls = class_names[np.argmax(preds[i])]
            conf = np.max(preds[i]) * 100
            print(f"  True: {true_cls:35s} | Pred: {pred_cls:35s} | Conf: {conf:.1f}%")


if __name__ == '__main__':
    main()
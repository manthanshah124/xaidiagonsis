#!/usr/bin/env python3
"""
Training script for XAI-PlantDoc.

Trains MobileNetV2 (and optionally EfficientNetB0) on the plant disease dataset.
Two-stage training: freeze base -> train head, then unfreeze top layers -> fine-tune.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.augmentation import get_augmentation_layers

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'model_name': 'MobileNetV2',
    'image_size': (224, 224),
    'num_classes': 14,
    'dropout_rate': 0.3,
    'last_conv_layer_name': 'Conv_1',  # MobileNetV2 last conv layer

    # Training - Stage 1 (frozen base)
    'stage1_epochs': 15,
    'stage1_lr': 1e-3,
    'stage1_batch_size': 32,

    # Training - Stage 2 (fine-tuning)
    'stage2_epochs': 40,
    'stage2_lr': 5e-5,
    'stage2_batch_size': 16,
    'fine_tune_layers': 100,  # Number of top layers to unfreeze

    # Callbacks
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,

    # Paths
    'processed_dir': PROJECT_ROOT / 'ml' / 'data' / 'processed',
    'models_dir': PROJECT_ROOT / 'ml' / 'models',
    'reports_dir': PROJECT_ROOT / 'ml' / 'reports',

    # Other
    'seed': 42,
    'class_weight_threshold': 50.0,  # Max class weight multiplier (was 10.0)
}

# Selected classes (must match prepare_dataset.py)
SELECTED_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Apple_Apple_scab",
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Apple_healthy",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(SELECTED_CLASSES)}


# =============================================================================
# Data Loading
# =============================================================================

def create_dataset(
    split: str,
    batch_size: int,
    image_size: Tuple[int, int],
    augment: bool = False,
    seed: int = 42,
    class_names: List[str] = None
) -> tf.data.Dataset:
    """Create tf.data.Dataset from directory structure."""
    processed_dir = DEFAULT_CONFIG['processed_dir']
    split_dir = processed_dir / split

    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    # Get class directories
    if class_names is None:
        class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    # Create dataset using image_dataset_from_directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=augment,
        seed=seed,
        interpolation='bilinear',
    )

    # Normalize to [-1, 1] for MobileNetV2
    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image, label

    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation if training
    if augment:
        augmentation = get_augmentation_layers(image_size=image_size, seed=seed)
        def augment_fn(image, label):
            return augmentation(image, training=True), label
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compute_class_weights(dataset: tf.data.Dataset, num_classes: int) -> Dict[int, float]:
    """Compute class weights from dataset."""
    labels = []
    for _, label_batch in dataset.unbatch():
        labels.append(label_batch.numpy())

    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )

    # Cap extreme weights
    class_weights = np.clip(class_weights, 1.0 / DEFAULT_CONFIG['class_weight_threshold'],
                           DEFAULT_CONFIG['class_weight_threshold'])

    return {i: float(w) for i, w in enumerate(class_weights)}


# =============================================================================
# Model Building
# =============================================================================

def build_model(
    model_name: str = 'MobileNetV2',
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 14,
    dropout_rate: float = 0.3,
    weights: str = 'imagenet'
) -> Tuple[tf.keras.Model, str]:
    """Build the model with pretrained base + custom head."""
    input_shape = (*image_size, 3)

    if model_name == 'MobileNetV2':
        base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None
        )
        # MobileNetV2 last conv layer is inside the base model
        last_conv_layer_name = 'out_relu'

    elif model_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None
        )
        last_conv_layer_name = 'top_conv'

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Build head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs, outputs, name=model_name)

    return model, last_conv_layer_name


def find_last_conv_layer(model: tf.keras.Model, model_name: str) -> str:
    """Find the last convolutional layer in the base model."""
    # Find base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name in ['MobileNetV2', 'EfficientNetB0']:
            base_model = layer
            break

    if base_model is None:
        # Try to find by checking layer types
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                break

    if base_model is None:
        # Fallback: return common names
        return 'out_relu' if model_name == 'MobileNetV2' else 'top_conv'

    # Find the last Conv2D layer in base model
    conv_layers = []
    for layer in base_model.layers:
        if 'Conv' in layer.__class__.__name__ or 'DepthwiseConv' in layer.__class__.__name__:
            conv_layers.append(layer.name)

    if conv_layers:
        return conv_layers[-1]

    # Fallback
    return 'out_relu' if model_name == 'MobileNetV2' else 'top_conv'


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile model with optimizer, loss, and metrics."""
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    return model


def get_callbacks(
    model_path: Path,
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7
) -> List[callbacks.Callback]:
    """Get training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        callbacks.CSVLogger(str(DEFAULT_CONFIG['reports_dir'] / 'training_log.csv'))
    ]


# =============================================================================
# Training Stages
# =============================================================================

def train_stage1(
    config: Dict,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float]
) -> tf.keras.callbacks.History:
    """Stage 1: Train head with frozen base."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training head (frozen base)")
    print("=" * 60)

    model, last_conv_layer = build_model(
        model_name=config['model_name'],
        image_size=config['image_size'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    )

    # Verify last conv layer exists (search in base model)
    last_conv_layer = find_last_conv_layer(model, config['model_name'])
    print(f"Detected last conv layer: {last_conv_layer}")

    # Freeze base
    for layer in model.layers:
        if 'MobileNetV2' in layer.name or 'EfficientNetB0' in layer.name:
            layer.trainable = False

    model = compile_model(model, config['stage1_lr'])

    print(f"Trainable params: {model.count_params()} (total)")
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable params: {trainable_count}")

    callbacks_list = get_callbacks(
        config['models_dir'] / f"{config['model_name']}_stage1.keras",
        config['early_stopping_patience'],
        config['reduce_lr_patience'],
        config['reduce_lr_factor'],
        config['min_lr']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['stage1_epochs'],
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save stage 1 model
    stage1_path = config['models_dir'] / f"{config['model_name']}_stage1.keras"
    model.save(stage1_path)
    print(f"Stage 1 model saved to {stage1_path}")

    return history, model, last_conv_layer


def train_stage2(
    config: Dict,
    model: tf.keras.Model,
    last_conv_layer: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float]
) -> tf.keras.callbacks.History:
    """Stage 2: Fine-tune top layers of base model."""
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning top layers")
    print("=" * 60)

    # Find base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name in ['MobileNetV2', 'EfficientNetB0']:
            base_model = layer
            break

    if base_model is None:
        # Try to find by checking layer types
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 50:
                base_model = layer
                break

    if base_model is None:
        raise ValueError("Could not find base model for fine-tuning")

    # Unfreeze top N layers
    fine_tune_from = len(base_model.layers) - config['fine_tune_layers']
    for layer in base_model.layers[fine_tune_from:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # Keep BatchNorm layers frozen
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model = compile_model(model, config['stage2_lr'])

    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable params after unfreezing: {trainable_count}")

    callbacks_list = get_callbacks(
        config['models_dir'] / f"{config['model_name']}_final.keras",
        config['early_stopping_patience'],
        config['reduce_lr_patience'],
        config['reduce_lr_factor'],
        config['min_lr']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['stage2_epochs'],
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save final model
    final_path = config['models_dir'] / f"{config['model_name']}_final.keras"
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    return history, model


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train plant disease classifier')
    parser.add_argument('--model', choices=['MobileNetV2', 'EfficientNetB0'], default='MobileNetV2')
    parser.add_argument('--stage1-epochs', type=int, default=10)
    parser.add_argument('--stage2-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fine-tune-layers', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Update config from args
    config = DEFAULT_CONFIG.copy()
    config['model_name'] = args.model
    config['stage1_epochs'] = args.stage1_epochs
    config['stage2_epochs'] = args.stage2_epochs
    config['stage1_batch_size'] = args.batch_size
    config['stage2_batch_size'] = max(8, args.batch_size // 2)
    config['stage1_lr'] = args.lr
    config['stage2_lr'] = args.lr * 0.01
    config['fine_tune_layers'] = args.fine_tune_layers
    config['seed'] = args.seed

    # Set seeds
    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create directories
    config['models_dir'].mkdir(parents=True, exist_ok=True)
    config['reports_dir'].mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config['reports_dir'] / 'train_config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    print(f"Training {config['model_name']} with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create datasets
    print("\nCreating datasets...")
    train_ds = create_dataset(
        'train', config['stage1_batch_size'], config['image_size'],
        augment=True, seed=config['seed'], class_names=SELECTED_CLASSES
    )
    val_ds = create_dataset(
        'val', config['stage1_batch_size'], config['image_size'],
        augment=False, seed=config['seed'], class_names=SELECTED_CLASSES
    )
    test_ds = create_dataset(
        'test', config['stage1_batch_size'], config['image_size'],
        augment=False, seed=config['seed'], class_names=SELECTED_CLASSES
    )

    # Compute class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(train_ds, config['num_classes'])
    print("Class weights:", {SELECTED_CLASSES[k]: v for k, v in class_weights.items()})

    # Stage 1
    stage1_history, model, last_conv_layer = train_stage1(
        config, train_ds, val_ds, class_weights
    )

    # Stage 2
    stage2_history, model = train_stage2(
        config, model, last_conv_layer, train_ds, val_ds, class_weights
    )

    # Combine histories
    combined_history = {
        'loss': stage1_history.history['loss'] + stage2_history.history['loss'],
        'accuracy': stage1_history.history['accuracy'] + stage2_history.history['accuracy'],
        'val_loss': stage1_history.history['val_loss'] + stage2_history.history['val_loss'],
        'val_accuracy': stage1_history.history['val_accuracy'] + stage2_history.history['val_accuracy'],
        'top3_acc': stage1_history.history.get('top3_acc', []) + stage2_history.history.get('top3_acc', []),
        'val_top3_acc': stage1_history.history.get('val_top3_acc', []) + stage2_history.history.get('val_top3_acc', []),
    }

    # Save final model as the main model
    final_model_path = config['models_dir'] / 'plant_model.keras'
    model.save(final_model_path)
    print(f"\nMain model saved to {final_model_path}")

    # Save class names
    class_names_path = config['models_dir'] / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(SELECTED_CLASSES, f, indent=2)
    print(f"Class names saved to {class_names_path}")

    # Save last conv layer name
    with open(config['models_dir'] / 'gradcam_layer.txt', 'w') as f:
        f.write(last_conv_layer)
    print(f"Grad-CAM layer name saved: {last_conv_layer}")

    # Save combined history
    history_path = config['reports_dir'] / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types
        serializable_history = {k: [float(v) for v in vals] for k, vals in combined_history.items()}
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    test_results = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    if len(test_results) > 2:
        print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")

    print("\nTraining complete!")
    return model, combined_history, test_results


if __name__ == '__main__':
    main()
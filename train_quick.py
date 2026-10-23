#!/usr/bin/env python3
"""
Quick training script for XAI-PlantDoc using a SMALL subset of data.
For rapid testing/debugging - uses only ~100-200 images per class.
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
from sklearn.utils.class_weight import compute_class_weight
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.augmentation import get_augmentation_layers

# =============================================================================
# Quick Config - Small subset for fast testing
# =============================================================================

QUICK_CONFIG = {
    'model_name': 'MobileNetV2',
    'image_size': (224, 224),
    'num_classes': 14,
    'dropout_rate': 0.3,
    
    # Training - Stage 1 (frozen base)
    'stage1_epochs': 5,
    'stage1_lr': 1e-3,
    'stage1_batch_size': 16,
    
    # Training - Stage 2 (fine-tuning)
    'stage2_epochs': 10,
    'stage2_lr': 5e-5,
    'stage2_batch_size': 16,
    'fine_tune_layers': 50,
    
    # Callbacks - more aggressive for quick test
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    
    # Paths
    'processed_dir': Path(__file__).parent.parent / 'ml' / 'data' / 'processed',
    'models_dir': Path(__file__).parent.parent / 'ml' / 'models',
    'reports_dir': Path(__file__).parent.parent / 'ml' / 'reports',
    
    # Quick mode: limit images per class
    'max_images_per_class': 150,
    'seed': 42,
    'class_weight_threshold': 20.0,
}

SELECTED_CLASSES = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus", "Apple_Apple_scab", "Apple_Black_rot",
    "Apple_Cedar_apple_rust", "Apple_healthy",
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def create_quick_dataset(processed_dir: Path, split: str, max_per_class: int = 150) -> Path:
    """Create a small subset dataset symlink for quick training."""
    src_dir = Path(__file__).parent.parent / 'ml' / 'data' / 'processed' / split
    dst_dir = Path(__file__).parent.parent / 'ml' / 'data' / 'processed_quick' / split
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for class_dir in (Path(__file__).parent.parent / 'ml' / 'data' / 'processed' / split).iterdir():
        if not class_dir.is_dir() or class_dir.name not in SELECTED_CLASSES:
            continue
            
        dst_class_dir = dst_dir / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get first N images
        images = list(class_dir.glob('*.jpg'))[:150]
        for img in images:
            shutil.copy2(img, dst_class_dir / img.name)
    
    return dst_dir


def create_datasets(config: Dict, quick_dir: Path):
    """Create datasets from quick subset."""
    
    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image, label
    
    def augment_fn(image, label):
        augmentation = get_augmentation_layers(image_size=IMAGE_SIZE, seed=config['seed'])
        return augmentation(image, training=True), label
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config['processed_dir'] / 'train',
        labels='inferred', label_mode='int', class_names=SELECTED_CLASSES,
        color_mode='rgb', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
        shuffle=True, seed=42, interpolation='bilinear'
    ).map(lambda x, y: (x / 127.5 - 1.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config['processed_dir'] / 'val',
        labels='inferred', label_mode='int', class_names=SELECTED_CLASSES,
        color_mode='rgb', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
        shuffle=False, seed=42, interpolation='bilinear'
    ).map(lambda x, y: (x / 127.5 - 1.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config['processed_dir'] / 'test',
        labels='inferred', label_mode='int', class_names=SELECTED_CLASSES,
        color_mode='rgb', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
        shuffle=False, seed=42, interpolation='bilinear'
    ).map(lambda x, y: (x / 127.5 - 1.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation to training
    augmentation = get_augmentation_layers(image_size=IMAGE_SIZE, seed=config['seed'])
    def augment_fn(image, label):
        return augmentation(image, training=True), label
    
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def compute_class_weights(dataset: tf.data.Dataset, num_classes: int, threshold: float = 20.0) -> Dict[int, float]:
    labels = []
    for _, label_batch in dataset.unbatch():
        labels.append(label_batch.numpy())
    labels = np.array(labels)
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
    class_weights = np.clip(class_weights, 1.0 / threshold, threshold)
    return {i: float(w) for i, w in enumerate(class_weights)}


def build_model(model_name: str = 'MobileNetV2', image_size: Tuple[int, int] = (224, 224), 
                num_classes: int = 14, dropout_rate: float = 0.3, weights: str = 'imagenet'):
    input_shape = (*image_size, 3)
    base_model = applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights='imagenet', pooling=None)
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    return model, 'Conv_1'


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')])
    return model


def find_last_conv_layer(model: tf.keras.Model, model_name: str) -> str:
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name in ['MobileNetV2', 'EfficientNetB0']:
            base_model = layer
            break
    if base_model:
        conv_layers = [l.name for l in base_model.layers if 'Conv' in l.__class__.__name__ or 'DepthwiseConv' in l.__class__.__name__]
        if conv_layers:
            return conv_layers[-1]
    return 'Conv_1'


def get_callbacks(model_path: Path, patience: int = 5, reduce_lr_patience: int = 3, 
                  reduce_lr_factor: float = 0.5, min_lr: float = 1e-7):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(model_path), monitor='val_loss', save_best_only=True, verbose=1),
    ]


def main():
    parser = argparse.ArgumentParser(description='Quick training for testing')
    parser.add_argument('--epochs', type=int, default=5, help='Stage 1 epochs')
    parser.add_argument('--stage2-epochs', type=int, default=10, help='Stage 2 epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--fine-tune-layers', type=int, default=50, help='Layers to unfreeze')
    args = parser.parse_args()
    
    config = QUICK_CONFIG.copy()
    config['stage1_epochs'] = args.epochs
    config['stage2_epochs'] = args.stage2_epochs
    config['stage1_batch_size'] = args.batch_size
    config['stage2_batch_size'] = args.batch_size
    config['stage1_lr'] = args.lr
    config['stage2_lr'] = args.lr * 0.01
    config['fine_tune_layers'] = args.fine_tune_layers
    
    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])
    
    config['models_dir'].mkdir(parents=True, exist_ok=True)
    config['reports_dir'].mkdir(parents=True, exist_ok=True)
    
    print(f"Quick training with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create datasets from full processed dir (use existing processed data)
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_datasets(QUICK_CONFIG, None)
    
    print("Computing class weights...")
    class_weights = compute_class_weights(train_ds, QUICK_CONFIG['num_classes'], QUICK_CONFIG['class_weight_threshold'])
    print("Class weights:", {SELECTED_CLASSES[k]: v for k, v in class_weights.items()})
    
    # Stage 1
    print("\n" + "=" * 60)
    print("STAGE 1: Training head (frozen base)")
    print("=" * 60)
    
    model, last_conv_layer = build_model()
    for layer in model.layers:
        if 'MobileNetV2' in layer.name:
            layer.trainable = False
    model = compile_model(model, config['stage1_lr'])
    
    callbacks_list = get_callbacks(
        config['models_dir'] / "plant_model_quick.keras",
        config['early_stopping_patience'], config['reduce_lr_patience'],
        config['reduce_lr_factor'], config['min_lr']
    )
    
    stage1_history = model.fit(
        train_ds, validation_data=val_ds, epochs=config['stage1_epochs'],
        class_weight=class_weights, callbacks=callbacks_list, verbose=1
    )
    
    # Stage 2
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning top layers")
    print("=" * 60)
    
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == 'MobileNetV2':
            base_model = layer
            break
    
    fine_tune_from = len(base_model.layers) - config['fine_tune_layers']
    for layer in base_model.layers[fine_tune_from:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    model = compile_model(model, config['stage2_lr'])
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable params after unfreezing: {trainable_count}")
    
    callbacks_list = get_callbacks(
        config['models_dir'] / "plant_model_quick.keras",
        config['early_stopping_patience'], config['reduce_lr_patience'],
        config['reduce_lr_factor'], config['min_lr']
    )
    
    stage2_history = model.fit(
        train_ds, validation_data=val_ds, epochs=config['stage2_epochs'],
        class_weight=class_weights, callbacks=callbacks_list, verbose=1
    )
    
    # Save final model
    final_path = config['models_dir'] / 'plant_model_quick.keras'
    model.save(final_path)
    print(f"\nQuick model saved to {final_path}")
    
    # Save class names
    class_names_path = config['models_dir'] / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(SELECTED_CLASSES, f, indent=2)
    
    # Save gradcam layer
    with open(config['models_dir'] / 'gradcam_layer.txt', 'w') as f:
        f.write('Conv_1')
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    if len(test_results) > 2:
        print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
    
    print("\nQuick training complete!")


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks, applications
    from sklearn.utils.class_weight import compute_class_weight
    from typing import Dict, Tuple, List
    import numpy as np
    import shutil
    from pathlib import Path
    import json
    import argparse
    
    main()
#!/usr/bin/env python3
"""
Evaluation script for XAI-PlantDoc.

Evaluates trained model on test set AND test_field (PlantDoc) set.
Generates:
- metrics.json (accuracy, per-class precision/recall/F1, macro/weighted F1)
- confusion_matrix.png
- training_curves.png (if history available)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from predictor.services.preprocessing import preprocess_for_model

# Paths
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

# Selected classes (must match training)
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
IDX_TO_CLASS = {i: cls for i, cls in enumerate(SELECTED_CLASSES)}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def create_dataset(split: str, batch_size: int = BATCH_SIZE, augment: bool = False) -> tf.data.Dataset:
    """Create tf.data.Dataset from directory structure."""
    split_dir = PROCESSED_DIR / split

    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=42,
        interpolation='bilinear',
    )

    # Normalize to [-1, 1] for MobileNetV2
    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image, label

    dataset = dataset.map(
        lambda x, y: (x / 127.5 - 1.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return dataset


def load_model_and_classes(model_path: Path = None, class_names_path: Path = None):
    """Load trained model and class names."""
    if model_path is None:
        model_path = MODELS_DIR / "plant_model.keras"
    if class_names_path is None:
        class_names_path = MODELS_DIR / "class_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names not found at {class_names_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    return model, class_names


def evaluate_on_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str],
    split_name: str
) -> Dict:
    """Evaluate model on a dataset and return detailed metrics."""
    print(f"\nEvaluating on {split_name}...")

    # Collect predictions
    y_true = []
    y_pred = []
    y_pred_probs = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.vstack(y_pred_probs)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(SELECTED_CLASSES)), zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=SELECTED_CLASSES,
        labels=range(len(SELECTED_CLASSES)),
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(SELECTED_CLASSES)))

    # Per-class accuracy
    per_class_acc = {}
    for i, cls in enumerate(SELECTED_CLASSES):
        mask = (y_true == i)
        if mask.sum() > 0:
            per_class_acc[cls] = float((y_pred[mask] == i).mean())
        else:
            per_class_acc[cls] = 0.0

    results = {
        'split': split_name,
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            cls: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
                'accuracy': per_class_acc[cls]
            }
            for i, cls in enumerate(SELECTED_CLASSES)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'num_samples': int(len(y_true))
    }

    # Print summary
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    for cls in SELECTED_CLASSES:
        m = results['per_class'][cls]
        print(f"  {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    # Normalize for display
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {output_path}")


def plot_training_curves(history_path: Path, output_path: Path):
    """Plot training curves from history JSON."""
    if not history_path.exists():
        print(f"  History file not found: {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss', color='#166534', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='#dc2626', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train Acc', color='#166534', linewidth=2)
    axes[0, 1].plot(history['val_accuracy'], label='Val Acc', color='#dc2626', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Top-3 Accuracy
    if 'top3_acc' in history and history['top3_acc']:
        axes[1, 0].plot(history['top3_acc'], label='Train Top-3', color='#166534', linewidth=2)
        axes[1, 0].plot(history['val_top3_acc'], label='Val Top-3', color='#dc2626', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate (if available)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5, 'Training completed', fontsize=12, ha='left', va='center')

    plt.suptitle('Training Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {output_path}")


def save_metrics(metrics: Dict, output_path: Path):
    """Save metrics to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"  Saved metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate XAI-PlantDoc model')
    parser.add_argument('--model', type=str, default=None, help='Path to model (.keras)')
    parser.add_argument('--class-names', type=str, default=None, help='Path to class_names.json')
    parser.add_argument('--history', type=str, default=None, help='Path to training_history.json')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for reports')
    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model) if args.model else MODELS_DIR / "plant_model.keras"
    class_names_path = Path(args.class_names) if args.class_names else MODELS_DIR / "class_names.json"
    history_path = Path(args.history) if args.history else REPORTS_DIR / "training_history.json"
    output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("XAI-PlantDoc Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Class names: {class_names_path}")
    print(f"Output dir: {output_dir}")

    # Load model and class names
    model, class_names = load_model_and_classes(model_path, class_names_path)
    print(f"Loaded model with {len(class_names)} classes: {class_names}")

    # Load training history (optional)
    history = None
    if args.history and Path(args.history).exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

    # Evaluate on test set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test",
        labels='inferred',
        label_mode='int',
        class_names=SELECTED_CLASSES,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=False,
        seed=42,
        interpolation='bilinear',
    ).map(lambda x, y: (x / 127.5 - 1.0, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    test_results = evaluate_on_dataset(model, test_ds, SELECTED_CLASSES, "test")

    # Evaluate on test_field (PlantDoc) if exists
    test_field_ds = None
    test_field_path = PROCESSED_DIR / "test_field"
    test_field_results = None
    
    if test_field_path.exists():
        try:
            test_field_ds = tf.keras.utils.image_dataset_from_directory(
                PROCESSED_DIR / "test_field",
                labels='inferred',
                label_mode='int',
                class_names=SELECTED_CLASSES,
                color_mode='rgb',
                batch_size=32,
                image_size=(224, 224),
                shuffle=False,
                seed=42,
                interpolation='bilinear',
            ).map(lambda x, y: (x / 127.5 - 1.0, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
            
            test_field_results = evaluate_on_dataset(model, test_field_ds, SELECTED_CLASSES, "test_field")
        except Exception as e:
            print(f"Warning: Could not evaluate test_field: {e}")
    else:
        print("Note: test_field directory not found, skipping field evaluation")

    # Compile all metrics
    all_metrics = {
        'test': test_results,
        'test_field': test_field_results,
        'model_info': {
            'model_path': str(model_path),
            'num_classes': len(SELECTED_CLASSES),
            'class_names': SELECTED_CLASSES,
            'input_shape': [224, 224, 3],
            'preprocessing': 'MobileNetV2 [-1, 1] normalization'
        }
    }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    save_metrics(all_metrics, metrics_path)

    # Plot confusion matrices
    plot_confusion_matrix(
        np.array(test_results['confusion_matrix']),
        SELECTED_CLASSES,
        output_dir / "confusion_matrix_test.png",
        "Confusion Matrix - Test Set (PlantVillage)"
    )

    if test_field_results:
        plot_confusion_matrix(
            np.array(test_field_results['confusion_matrix']),
            SELECTED_CLASSES,
            output_dir / "confusion_matrix_test_field.png",
            "Confusion Matrix - Test Field (PlantDoc)"
        )

    # Plot training curves
    if args.history and Path(args.history).exists():
        plot_training_curves(Path(args.history), output_dir / "training_curves.png")

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Macro F1: {test_results['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_results['weighted_f1']:.4f}")
    if test_field_results:
        print(f"Test Field Accuracy: {test_field_results['accuracy']:.4f}")
        print(f"Test Field Macro F1: {test_field_results['macro_f1']:.4f}")
    print(f"\nReports saved to: {output_dir}")
    print(f"  - metrics.json")
    print(f"  - confusion_matrix_test.png")
    if test_field_results:
        print(f"  - confusion_matrix_test_field.png")
    if args.history:
        print(f"  - training_curves.png")


if __name__ == '__main__':
    import tensorflow as tf
    main()
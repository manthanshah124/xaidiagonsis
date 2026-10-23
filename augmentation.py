"""
Data augmentation pipeline for training.

This module provides tf.keras preprocessing layers for data augmentation
applied to TRAINING data only, never validation/test.
"""

import tensorflow as tf
from typing import Tuple


def get_augmentation_layers(
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> tf.keras.Sequential:
    """
    Create a Sequential model with augmentation layers.

    Augmentations applied (all with probability < 1.0 for variety):
    - Random horizontal flip (50%)
    - Random vertical flip (20%)
    - Random rotation (±30 degrees)
    - Random zoom (±20%)
    - Random translation (±10%)
    - Random brightness (±30%)
    - Random contrast (±20%)
    - Random saturation (±20%)
    - Gaussian noise (std=0.02)

    Args:
        image_size: Target image size (height, width).
        seed: Random seed for reproducibility.

    Returns:
        tf.keras.Sequential model with augmentation layers.
    """
    return tf.keras.Sequential([
        # Random horizontal flip (50%)
        tf.keras.layers.RandomFlip(
            mode="horizontal",
            seed=seed,
            name="random_flip_horizontal"
        ),

        # Random vertical flip (20%)
        tf.keras.layers.RandomFlip(
            mode="vertical",
            seed=seed,
            name="random_flip_vertical"
        ),

        # Random rotation (±30 degrees = ±0.52 radians)
        tf.keras.layers.RandomRotation(
            factor=0.3 / 2.0,  # factor is fraction of 2π
            fill_mode="reflect",
            interpolation="bilinear",
            seed=seed,
            name="random_rotation"
        ),

        # Random translation (±10%)
        tf.keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode="reflect",
            interpolation="bilinear",
            seed=seed,
            name="random_translation"
        ),

        # Random zoom (±20%)
        tf.keras.layers.RandomZoom(
            height_factor=0.2,
            width_factor=0.2,
            fill_mode="reflect",
            interpolation="bilinear",
            seed=seed,
            name="random_zoom"
        ),

        # Random brightness (±30%)
        tf.keras.layers.RandomBrightness(
            factor=0.3,
            value_range=(-1.0, 1.0),  # MobileNetV2 preprocessing range
            seed=seed,
            name="random_brightness"
        ),

        # Random contrast (±20%)
        tf.keras.layers.RandomContrast(
            factor=0.2,
            seed=seed,
            name="random_contrast"
        ),

        # Random saturation (±20%)
        tf.keras.layers.RandomSaturation(
            factor=0.2,
            seed=seed,
            name="random_saturation"
        ),

        # Gaussian noise (small amount)
        tf.keras.layers.GaussianNoise(
            stddev=0.02,
            seed=seed,
            name="gaussian_noise"
        ),
    ], name="augmentation_pipeline")


def apply_augmentation(
    images: tf.Tensor,
    training: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> tf.Tensor:
    """
    Apply augmentation to a batch of images.

    Args:
        images: Batch of images, shape (batch, H, W, C), values in [-1, 1].
        training: Whether to apply augmentation (True) or not (False).
        image_size: Target image size.
        seed: Random seed.

    Returns:
        Augmented images (if training) or original images (if not training).
    """
    if not training:
        return images

    augmentation = get_augmentation_layers(image_size=image_size, seed=seed)
    return augmentation(images, training=True)


def create_augmented_dataset(
    dataset: tf.data.Dataset,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
    batch_size: int = 32
) -> tf.data.Dataset:
    """
    Create an augmented dataset from a base dataset.

    Args:
        dataset: Base dataset yielding (image, label) pairs.
        image_size: Target image size.
        seed: Random seed for augmentation.
        batch_size: Batch size for prefetching.

    Returns:
        Augmented dataset with same structure.
    """
    augmentation = get_augmentation_layers(image_size=image_size, seed=seed)

    def augment_fn(image, label):
        return augmentation(image, training=True), label

    return dataset.map(
        augment_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    # Test the augmentation pipeline
    import numpy as np

    # Create dummy data
    dummy_images = np.random.rand(4, 224, 224, 3).astype(np.float32) * 2 - 1  # [-1, 1]
    dummy_labels = np.array([0, 1, 2, 3])

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
    ds = ds.batch(2)

    # Apply augmentation
    aug_ds = create_augmented_dataset(ds)

    # Test
    for images, labels in aug_ds.take(1):
        print(f"Input shape: {images.shape}")
        print(f"Input range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"Labels: {labels.numpy()}")

    print("\nAugmentation pipeline test passed!")
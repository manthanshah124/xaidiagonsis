#!/usr/bin/env python3
"""
Dataset preparation script for XAI-PlantDoc.

This script:
1. Selects configurable crops/classes from PlantVillage dataset
2. Organizes images into class-wise folders
3. Skips corrupted/unreadable images and logs them
4. Resizes to 224x224 and saves to ml/data/processed/{train,val,test}/<class>/
5. Splits 70/15/15 stratified by class with fixed random seed
6. Detects near-duplicates using perceptual hash and keeps duplicates in same split
7. Keeps PlantDoc (real field images) as separate evaluation set "test_field"
"""

import os
import sys
import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import random
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

# Selected classes for this run (Tomato + Apple = 14 classes)
SELECTED_CLASSES = [
    # Tomato classes (10)
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    # Apple classes (4)
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

# Class name mapping for output (cleaner names)
CLASS_NAME_MAP = {
    "Tomato___Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___healthy": "Tomato_healthy",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato_Spider_mites",
    "Tomato___Target_Spot": "Tomato_Target_Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Apple___Apple_scab": "Apple_Apple_scab",
    "Apple___Black_rot": "Apple_Black_rot",
    "Apple___Cedar_apple_rust": "Apple_Cedar_apple_rust",
    "Apple___healthy": "Apple_healthy",
}

# Reverse mapping for PlantDoc folder names
PLANTDOC_CLASS_MAP = {
    "Tomato_Bacterial_spot": "Tomato_leaf_bacterial_spot",
    "Tomato_Early_blight": "Tomato_Early_blight_leaf",
    "Tomato_healthy": "Tomato_leaf",
    "Tomato_Late_blight": "Tomato_leaf_late_blight",
    "Tomato_Leaf_Mold": "Tomato_mold_leaf",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites": "Tomato_two_spotted_spider_mites_leaf",
    "Tomato_Target_Spot": "Tomato_Target_Spot",  # May not exist in PlantDoc
    "Tomato_mosaic_virus": "Tomato_leaf_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato_leaf_yellow_virus",
    "Apple_Apple_scab": "Apple_Scab_Leaf",
    "Apple_Black_rot": "Apple_rust_leaf",  # Closest match
    "Apple_Cedar_apple_rust": "Apple_rust_leaf",  # Closest match
    "Apple_healthy": "Apple_leaf",
}

# Paths
RAW_PLANT_VILLAGE = PROJECT_ROOT / "ml" / "data" / "raw" / "PlantVillage dataset" / "color"
RAW_PLANT_DOC = PROJECT_ROOT / "ml" / "data" / "raw" / "PlantDoc Classification dataset"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Image settings
TARGET_SIZE = (224, 224)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}

# Perceptual hash threshold for near-duplicate detection
# Lower = more similar; 0 = identical; typical threshold 5-10 for near-duplicates
PHASH_THRESHOLD = 4

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageInfo:
    """Information about a single image."""
    path: Path
    class_name: str
    phash: Optional[str] = None
    is_valid: bool = True
    error: Optional[str] = None


@dataclass
class SplitStats:
    """Statistics for a dataset split."""
    train: Dict[str, int]
    val: Dict[str, int]
    test: Dict[str, int]
    test_field: Dict[str, int]
    corrupted_count: int
    total_images: int


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('prepare_dataset')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# Core Functions
# =============================================================================

def compute_phash(image_path: Path) -> Optional[str]:
    """Compute perceptual hash of an image."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize to small size for hashing
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            return str(imagehash.phash(img))
    except Exception:
        return None


def is_valid_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if image is valid and readable."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        # Re-open to check it can be loaded fully
        with Image.open(image_path) as img:
            img.load()
        return True, None
    except Exception as e:
        return False, str(e)


def collect_images(raw_dir: Path, selected_classes: List[str], logger: logging.Logger) -> List[ImageInfo]:
    """Collect all valid images from the raw directory for selected classes."""
    images = []
    corrupted = []

    for class_name in selected_classes:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        image_files = [f for f in class_dir.iterdir()
                      if f.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}]

        logger.info(f"Found {len(image_files)} images in {class_name}")

        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            is_valid, error = is_valid_image(img_path)
            if not is_valid:
                corrupted.append((str(img_path), error))
                images.append(ImageInfo(
                    path=img_path,
                    class_name=class_name,
                    is_valid=False,
                    error=error
                ))
                continue

            # Compute perceptual hash for near-duplicate detection
            phash = compute_phash(img_path)

            images.append(ImageInfo(
                path=img_path,
                class_name=class_name,
                phash=phash,
                is_valid=True
            ))

    if corrupted:
        logger.warning(f"Found {len(corrupted)} corrupted/unreadable images")
        for path, error in corrupted[:10]:
            logger.warning(f"  Corrupted: {path} - {error}")
        if len(corrupted) > 10:
            logger.warning(f"  ... and {len(corrupted) - 10} more")

    return images


def group_near_duplicates(images: List[ImageInfo], threshold: int = PHASH_THRESHOLD) -> List[List[int]]:
    """
    Group images by perceptual hash similarity.

    Returns list of groups, where each group is a list of indices into the images list.
    Images in the same group are near-duplicates and should be kept in the same split.
    """
    valid_images = [(i, img) for i, img in enumerate(images) if img.is_valid and img.phash is not None]

    if not valid_images:
        return [[i] for i in range(len(images))]

    # Build similarity graph
    n = len(valid_images)
    groups = []
    visited = set()

    for idx, (i, img) in enumerate(valid_images):
        if i in visited:
            continue

        group = [i]
        visited.add(i)
        phash1 = imagehash.hex_to_hash(img.phash)

        # Compare with all subsequent images
        for jdx in range(idx + 1, n):
            j, other_img = valid_images[jdx]
            if j in visited:
                continue

            phash2 = imagehash.hex_to_hash(other_img.phash)
            distance = phash1 - phash2

            if distance <= threshold:
                group.append(j)
                visited.add(j)

        groups.append(group)

    # Add invalid images as singleton groups
    invalid_indices = [i for i, img in enumerate(images) if not img.is_valid or img.phash is None]
    for idx in invalid_indices:
        groups.append([idx])

    return groups


def split_groups_stratified(
    groups: List[List[int]],
    images: List[ImageInfo],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = RANDOM_SEED
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split groups into train/val/test maintaining class stratification.

    Since groups contain near-duplicates, we keep entire groups together.
    We stratify by class to maintain class distribution across splits.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group by class
    class_groups = defaultdict(list)
    for group in groups:
        # Get class from first valid image in group
        class_name = None
        for idx in group:
            if images[idx].is_valid:
                class_name = images[idx].class_name
                break
        if class_name is None:
            class_name = "unknown"
        class_groups[class_name].append(group)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_name, class_group_list in class_groups.items():
        # Flatten groups for this class
        random.shuffle(class_group_list)
        flat_indices = [idx for group in class_group_list for idx in group]

        n = len(flat_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_indices.extend(flat_indices[:n_train])
        val_indices.extend(flat_indices[n_train:n_train + n_val])
        test_indices.extend(flat_indices[n_train + n_val:])

    return train_indices, val_indices, test_indices


def save_resized_image(src_path: Path, dst_path: Path, target_size: Tuple[int, int] = TARGET_SIZE) -> bool:
    """Resize and save image to target size."""
    try:
        with Image.open(src_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize with high-quality resampling
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            # Save as JPEG with high quality
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, 'JPEG', quality=95, optimize=True)
        return True
    except Exception as e:
        print(f"Error saving {src_path} to {dst_path}: {e}")
        return False


def copy_plantdoc_images(
    plantdoc_root: Path,
    class_map: Dict[str, str],
    output_dir: Path,
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Dict[str, int]:
    """
    Copy PlantDoc images to test_field directory, resizing to target size.

    Returns dict of class_name -> count.
    """
    counts = defaultdict(int)

    for class_name, plantdoc_folder in class_map.items():
        src_dir = plantdoc_root / "test" / plantdoc_folder
        if not src_dir.exists():
            # Try train folder if test doesn't have it
            src_dir = plantdoc_root / "train" / plantdoc_folder
        if not src_dir.exists():
            print(f"Warning: PlantDoc folder not found: {src_dir}")
            continue

        dst_dir = output_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in src_dir.iterdir()
                      if f.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}]

        for img_path in image_files:
            is_valid, _ = is_valid_image(img_path)
            if not is_valid:
                continue

            dst_path = dst_dir / f"{img_path.stem}.jpg"
            if save_resized_image(img_path, dst_path, target_size):
                counts[class_name] += 1

    return dict(counts)


def prepare_dataset():
    """Main dataset preparation function."""
    # Setup directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = REPORTS_DIR / "prepare_dataset.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Starting dataset preparation")
    logger.info(f"Selected classes: {SELECTED_CLASSES}")
    logger.info(f"Target size: {TARGET_SIZE}")
    logger.info(f"Split ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Perceptual hash threshold: {PHASH_THRESHOLD}")

    # Step 1: Collect images from PlantVillage
    logger.info("Collecting images from PlantVillage...")
    images = collect_images(RAW_PLANT_VILLAGE, SELECTED_CLASSES, logger)

    valid_images = [img for img in images if img.is_valid]
    corrupted_count = len(images) - len(valid_images)
    logger.info(f"Total images: {len(images)}, Valid: {len(valid_images)}, Corrupted: {corrupted_count}")

    if not valid_images:
        logger.error("No valid images found!")
        return

    # Step 2: Group near-duplicates
    logger.info("Computing perceptual hashes and grouping near-duplicates...")
    groups = group_near_duplicates(images)
    logger.info(f"Formed {len(groups)} groups from {len(valid_images)} valid images")

    # Log group statistics
    group_sizes = [len(g) for g in groups]
    logger.info(f"Group size stats: min={min(group_sizes)}, max={max(group_sizes)}, "
                f"mean={np.mean(group_sizes):.1f}, median={np.median(group_sizes):.1f}")

    # Step 3: Stratified split by groups
    logger.info("Performing stratified split...")
    train_idx, val_idx, test_idx = split_groups_stratified(
        groups, images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Step 4: Save processed images
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }

    split_counts = defaultdict(lambda: defaultdict(int))

    for split_name, indices in splits.items():
        logger.info(f"Saving {split_name} split ({len(indices)} images)...")
        for idx in tqdm(indices, desc=f"Saving {split_name}"):
            img_info = images[idx]
            if not img_info.is_valid:
                continue

            class_name = CLASS_NAME_MAP.get(img_info.class_name, img_info.class_name)
            dst_path = PROCESSED_DIR / split_name / class_name / f"{img_info.path.stem}.jpg"

            if save_resized_image(img_info.path, dst_path):
                split_counts[split_name][class_name] += 1

    # Step 5: Copy PlantDoc images as test_field
    logger.info("Copying PlantDoc images as test_field...")
    test_field_counts = copy_plantdoc_images(
        RAW_PLANT_DOC, PLANTDOC_CLASS_MAP, PROCESSED_DIR / "test_field"
    )
    split_counts['test_field'] = test_field_counts

    # Step 6: Save class_names.json
    class_names = [CLASS_NAME_MAP.get(c, c) for c in SELECTED_CLASSES]
    class_names_path = MODELS_DIR / "class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Saved class names to {class_names_path}")

    # Step 7: Generate report
    report = {
        'selected_classes': SELECTED_CLASSES,
        'class_name_map': CLASS_NAME_MAP,
        'split_counts': dict(split_counts),
        'corrupted_count': corrupted_count,
        'total_valid_images': len(valid_images),
        'num_groups': len(groups),
        'phash_threshold': PHASH_THRESHOLD,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO,
        },
        'random_seed': RANDOM_SEED,
        'target_size': TARGET_SIZE,
    }

    report_path = REPORTS_DIR / "dataset_preparation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved preparation report to {report_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    for split_name in ['train', 'val', 'test', 'test_field']:
        total = sum(split_counts[split_name].values())
        logger.info(f"{split_name}: {total} images")
        for class_name, count in sorted(split_counts[split_name].items()):
            logger.info(f"  {class_name}: {count}")

    return report


if __name__ == '__main__':
    prepare_dataset()
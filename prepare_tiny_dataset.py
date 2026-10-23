#!/usr/bin/env python3
"""
Create a tiny dataset with N images per class for quick testing.
Usage: python ml/prepare_tiny_dataset.py --images-per-class 3
"""

import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SELECTED_CLASSES = [
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
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

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

RAW_PLANT_VILLAGE = PROJECT_ROOT / "ml" / "data" / "raw" / "PlantVillage dataset" / "color"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed_tiny"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

TARGET_SIZE = (224, 224)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
RANDOM_SEED = 42

SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

random.seed(RANDOM_SEED)


def is_valid_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False


def save_resized_image(src_path: Path, dst_path: Path, target_size: tuple = TARGET_SIZE) -> bool:
    try:
        with Image.open(src_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, 'JPEG', quality=95, optimize=True)
        return True
    except Exception as e:
        print(f"Error saving {src_path}: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create tiny dataset')
    parser.add_argument('--images-per-class', type=int, default=3, help='Images per class (default: 3)')
    parser.add_argument('--output-dir', type=str, default='processed_tiny', help='Output directory name')
    args = parser.parse_args()

    IMAGES_PER_CLASS = args.images_per_class
    output_dir = PROJECT_ROOT / "ml" / "data" / args.output_dir

    print(f"Creating tiny dataset: {IMAGES_PER_CLASS} images/class × {len(SELECTED_CLASSES)} classes")
    print(f"Output: {output_dir}")

    # Clean output
    if output_dir.exists():
        shutil.rmtree(output_dir)

    class_names = []
    split_counts = defaultdict(lambda: defaultdict(int))

    for class_name in SELECTED_CLASSES:
        src_dir = RAW_PLANT_VILLAGE / class_name
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found")
            continue

        # Get valid images
        image_files = [f for f in src_dir.iterdir()
                      if f.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}]
        valid_files = [f for f in image_files if is_valid_image(f)]

        if len(valid_files) < IMAGES_PER_CLASS:
            print(f"Warning: {class_name} has only {len(valid_files)} valid images (need {IMAGES_PER_CLASS})")
            selected = valid_files
        else:
            selected = random.sample(valid_files, IMAGES_PER_CLASS)

        clean_name = CLASS_NAME_MAP.get(class_name, class_name)
        class_names.append(clean_name)

        # Split selected images - ensure at least 1 per split
        n = len(selected)
        n_train = max(1, int(n * SPLIT_RATIOS['train']))
        n_val = max(1, int(n * SPLIT_RATIOS['val']))
        n_test = max(1, n - n_train - n_val)
        # Adjust if total exceeds n
        while n_train + n_val + n_test > n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        random.shuffle(selected)
        splits = {
            'train': selected[:n_train],
            'val': selected[n_train:n_train + n_val],
            'test': selected[n_train + n_val:],
        }

        for split_name, files in splits.items():
            for img_path in files:
                dst_path = output_dir / split_name / clean_name / f"{img_path.stem}.jpg"
                if save_resized_image(img_path, dst_path):
                    split_counts[split_name][clean_name] += 1

    # Save class_names.json
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class_names.json ({len(class_names)} classes)")

    # Print summary
    print("\nDataset summary:")
    for split_name in ['train', 'val', 'test']:
        total = sum(split_counts[split_name].values())
        print(f"  {split_name}: {total} images")
        for cn, cnt in sorted(split_counts[split_name].items()):
            print(f"    {cn}: {cnt}")

    print(f"\nDone! Use this dataset by setting:")
    print(f"  PROCESSED_DIR = '{output_dir}'")


if __name__ == '__main__':
    main()
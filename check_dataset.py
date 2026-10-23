#!/usr/bin/env python3
"""
Dataset validation and check script for XAI-PlantDoc.

This script:
1. Prints class list and per-class image counts per split
2. Confirms zero filename/hash overlap between train and test
3. Flags heavy class imbalance
4. Saves report to ml/reports/dataset_report.json
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

SPLITS = ['train', 'val', 'test', 'test_field']

# Class imbalance threshold (if any class has < 10% of median count)
IMBALANCE_THRESHOLD = 0.1


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_image_files(split_dir: Path) -> List[Path]:
    """Get all image files in a split directory."""
    if not split_dir.exists():
        return []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = []
    for ext in extensions:
        files.extend(split_dir.rglob(f'*{ext}'))
        files.extend(split_dir.rglob(f'*{ext.upper()}'))
    return files


def collect_split_data(processed_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Collect all image files organized by split and class."""
    data = {}
    for split in SPLITS:
        split_dir = processed_dir / split
        if not split_dir.exists():
            data[split] = {}
            continue

        data[split] = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                files = get_image_files(class_dir)
                data[split][class_dir.name] = files
    return data


def compute_overlap(data: Dict[str, Dict[str, List[Path]]]) -> Dict[str, int]:
    """Check for filename and content hash overlap between train and test/val."""
    # Collect all filenames and hashes from train
    train_files = data.get('train', {})
    train_filenames = set()
    train_hashes = set()

    for class_files in train_files.values():
        for f in class_files:
            train_filenames.add(f.name)
            train_hashes.add(compute_file_hash(f))

    # Check other splits
    overlap = {
        'train_val_filename': 0,
        'train_val_hash': 0,
        'train_test_filename': 0,
        'train_test_hash': 0,
        'train_test_field_filename': 0,
        'train_test_field_hash': 0,
        'val_test_filename': 0,
        'val_test_hash': 0,
    }

    # Check val
    val_files = data.get('val', {})
    val_filenames = set()
    val_hashes = set()
    for class_files in val_files.values():
        for f in class_files:
            val_filenames.add(f.name)
            val_hashes.add(compute_file_hash(f))

    overlap['train_val_filename'] = len(train_filenames & val_filenames)
    overlap['train_val_hash'] = len(train_hashes & val_hashes)

    # Check test
    test_files = data.get('test', {})
    test_filenames = set()
    test_hashes = set()
    for class_files in test_files.values():
        for f in class_files:
            test_filenames.add(f.name)
            test_hashes.add(compute_file_hash(f))

    overlap['train_test_filename'] = len(train_filenames & test_filenames)
    overlap['train_test_hash'] = len(train_hashes & test_hashes)

    # Check val-test overlap
    overlap['val_test_filename'] = len(val_filenames & test_filenames)
    overlap['val_test_hash'] = len(val_hashes & test_hashes)

    # Check test_field
    test_field_files = data.get('test_field', {})
    test_field_filenames = set()
    test_field_hashes = set()
    for class_files in test_field_files.values():
        for f in class_files:
            test_field_filenames.add(f.name)
            test_field_hashes.add(compute_file_hash(f))

    overlap['train_test_field_filename'] = len(train_filenames & test_field_filenames)
    overlap['train_test_field_hash'] = len(train_hashes & test_field_hashes)

    return overlap


def check_class_balance(data: Dict[str, Dict[str, List[Path]]]) -> Dict:
    """Check for class imbalance in each split."""
    balance_report = {}

    for split in SPLITS:
        split_data = data.get(split, {})
        if not split_data:
            balance_report[split] = {'status': 'empty', 'classes': {}}
            continue

        class_counts = {cls: len(files) for cls, files in split_data.items()}
        counts = list(class_counts.values())

        if not counts:
            balance_report[split] = {'status': 'empty', 'classes': class_counts}
            continue

        median_count = sorted(counts)[len(counts) // 2]
        threshold = median_count * IMBALANCE_THRESHOLD

        imbalanced = {cls: count for cls, count in class_counts.items() if count < threshold}

        balance_report[split] = {
            'status': 'ok' if not imbalanced else 'imbalanced',
            'median_count': median_count,
            'threshold': threshold,
            'imbalanced_classes': imbalanced,
            'class_counts': class_counts,
        }

    return balance_report


def print_report(data: Dict, overlap: Dict, balance: Dict):
    """Print formatted report to console."""
    print("=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    # Class list and counts per split
    print("\n1. CLASS LIST AND COUNTS PER SPLIT")
    print("-" * 70)

    all_classes = set()
    for split in SPLITS:
        all_classes.update(data.get(split, {}).keys())

    sorted_classes = sorted(all_classes)

    # Header
    header = f"{'Class':<40} {'Train':>8} {'Val':>8} {'Test':>8} {'Test_Field':>10}"
    print(header)
    print("-" * 70)

    for cls in sorted_classes:
        row = f"{cls:<40}"
        for split in SPLITS:
            count = len(data.get(split, {}).get(cls, []))
            if split == 'test_field':
                row += f" {count:>10}"
            else:
                row += f" {count:>8}"
        print(row)

    # Totals
    print("-" * 70)
    total_row = f"{'TOTAL':<40}"
    for split in SPLITS:
        total = sum(len(files) for files in data.get(split, {}).values())
        if split == 'test_field':
            total_row += f" {total:>10}"
        else:
            total_row += f" {total:>8}"
    print(total_row)

    # Overlap check
    print("\n2. TRAIN/TEST/VAL OVERLAP CHECK")
    print("-" * 70)
    print("Filename overlap (should be 0):")
    print(f"  train & val:         {overlap['train_val_filename']:>5} filenames")
    print(f"  train & test:        {overlap['train_test_filename']:>5} filenames")
    print(f"  train & test_field:  {overlap['train_test_field_filename']:>5} filenames")
    print(f"  val & test:          {overlap['val_test_filename']:>5} filenames")

    print("\nContent hash overlap (should be 0):")
    print(f"  train & val:         {overlap['train_val_hash']:>5} images")
    print(f"  train & test:        {overlap['train_test_hash']:>5} images")
    print(f"  train & test_field:  {overlap['train_test_field_hash']:>5} images")
    print(f"  val & test:          {overlap['val_test_hash']:>5} images")

    # Class balance
    print("\n3. CLASS BALANCE CHECK")
    print("-" * 70)
    for split in SPLITS:
        if split not in balance:
            continue
        b = balance[split]
        if b['status'] == 'empty':
            print(f"  {split}: EMPTY")
            continue
        elif b['status'] == 'imbalanced':
            print(f"  {split}: [WARNING] IMBALANCED (median={b['median_count']}, threshold={b['threshold']:.1f})")
            for cls, count in b['imbalanced_classes'].items():
                print(f"    {cls}: {count} images ({count/b['median_count']*100:.1f}% of median)")
        else:
            print(f"  {split}: [OK] Balanced (median={b['median_count']})")

    # Summary
    print("\n4. SUMMARY")
    print("-" * 70)
    total_images = sum(
        sum(len(files) for files in data.get(split, {}).values())
        for split in SPLITS
    )
    total_train_test = sum(
        len(files) for files in data.get('train', {}).values()
    ) + sum(
        len(files) for files in data.get('test', {}).values()
    )

    all_clean = (
        overlap['train_val_filename'] == 0 and
        overlap['train_test_filename'] == 0 and
        overlap['train_val_hash'] == 0 and
        overlap['train_test_hash'] == 0
    )

    print(f"  Total images across all splits: {total_images}")
    print(f"  Train + Test images: {total_train_test}")
    print(f"  No train/val/test overlap: {'[PASS]' if all_clean else '[FAIL]'}")
    print(f"  Class names file exists: {'[OK]' if (MODELS_DIR / 'class_names.json').exists() else '[MISSING]'}")


def save_report(data: Dict, overlap: Dict, balance: Dict, output_path: Path):
    """Save detailed report to JSON."""
    # Convert Path objects to strings for JSON serialization
    serializable_data = {}
    for split, classes in data.items():
        serializable_data[split] = {}
        for cls, files in classes.items():
            serializable_data[split][cls] = [str(f) for f in files]

    report = {
        'splits': {
            split: {cls: len(files) for cls, files in classes.items()}
            for split, classes in data.items()
        },
        'overlap': overlap,
        'class_balance': balance,
        'total_images': sum(
            sum(len(files) for files in classes.values())
            for classes in data.values()
        ),
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    print("Checking dataset...")

    # Collect data
    data = collect_split_data(PROCESSED_DIR)

    # Check overlap
    print("Computing content hashes for overlap check...")
    overlap = compute_overlap(data)

    # Check class balance
    balance = check_class_balance(data)

    # Print report
    print_report(data, overlap, balance)

    # Save report
    report_path = REPORTS_DIR / "dataset_report.json"
    save_report(data, overlap, balance, report_path)
    print(f"\nDetailed report saved to: {report_path}")

    # Return exit code based on validation
    all_clean = (
        overlap['train_val_filename'] == 0 and
        overlap['train_test_filename'] == 0 and
        overlap['train_val_hash'] == 0 and
        overlap['train_test_hash'] == 0
    )

    if all_clean:
        print("\n[SUCCESS] All checks passed!")
        return 0
    else:
        print("\n[FAILED] Validation failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
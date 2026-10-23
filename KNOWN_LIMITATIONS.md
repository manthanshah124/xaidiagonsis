# Known Limitations

This document documents known limitations, bugs, and measured evaluation results for XAI-PlantDoc.

## Measured Evaluation Results

### Model Performance (Test Set - PlantVillage)
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 4.57% |
| Top-3 Accuracy | 15.34% |
| Test Loss | 2.6065 |
| Number of Classes | 14 |
| Test Samples | 3,214 |

### Model Performance (Test Field Set - PlantDoc)
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | TBD |
| Top-3 Accuracy | TBD |
| Number of Classes Tested | 13 (Tomato_Target_Spot missing in PlantDoc) |

### Training Details
- **Architecture**: MobileNetV2 (ImageNet pretrained) + GlobalAveragePooling2D + Dropout(0.3) + Dense(14)
- **Stage 1**: Frozen base, 8 epochs (early stopped at epoch 4), LR=1e-3, Batch=32
- **Stage 2**: Fine-tuned top 50 layers, 6 epochs (early stopped at epoch 1), LR=1e-5, Batch=16
- **Class Weights**: Applied (inverse frequency, capped at 10x)
- **Data Augmentation**: RandomFlip, RandomRotation(±20°), RandomZoom(±10%), RandomBrightness(±20%), RandomContrast(±10%)
- **Near-Duplicate Handling**: Perceptual hashing (pHash threshold=8), duplicates kept in same split

## Known Limitations

### 1. Very Low Test Accuracy (4.57%)
**Status**: Critical
**Cause**: The model fails to learn meaningful features. Possible causes:
- Insufficient fine-tuning (only 1 epoch of Stage 2 completed due to early stopping)
- Learning rate too low (1e-5) for fine-tuning
- Class imbalance despite weighting
- Domain gap between PlantVillage (lab) and real-world conditions
- Near-duplicate handling may not be aggressive enough (threshold=8)
- Only 1 epoch of Stage 2 training completed before early stopping

**Impact**: Model predictions are essentially random (random chance = 7.14% for 14 classes).

### 2. Closed-Set Classifier Behavior on Non-Leaf Images
**Status**: Expected Limitation (by design)
**Description**: The model is a closed-set classifier trained on 14 specific plant disease classes. When presented with non-leaf images (e.g., a photo of a dog, a building, a person), it will still predict one of the 14 classes with some confidence score.
**Mitigation**: Low-confidence warning threshold (default 60%) typically triggers for out-of-distribution images. Future stages will add image quality checks and uncertainty handling.

### 3. Grad-CAM Quality
**Status**: Functional but limited
**Description**: Grad-CAM overlays are generated but may not be meaningful due to:
- Low model accuracy means attention maps don't correspond to meaningful features
- Last conv layer (Conv_1) has 7×7 spatial resolution, leading to coarse heatmaps
- Model not trained well enough to learn localized features

### 4. Class Imbalance
**Status**: Partially addressed
**Description**: Significant class imbalance in training data (e.g., Tomato_Yellow_Leaf_Curl_Virus: 7,492 samples vs Tomato_mosaic_virus: 520 samples). Class weights applied (capped at 10x) but may be insufficient.

### 5. PlantDoc Field Set Missing Class
**Status**: Known gap
**Description**: `Tomato_Target_Spot` class exists in PlantVillage but not in PlantDoc Classification dataset. Cannot evaluate field performance for this class.

### 6. Near-Duplicate Detection Threshold
**Status**: May need tuning
**Description**: pHash threshold of 8 may not catch all near-duplicates. Visual inspection suggests some near-duplicates still exist across splits.

### 7. MobileNetV2 Last Conv Layer Resolution
**Status**: Architectural limitation
**Description**: MobileNetV2's last conv layer (Conv_1) has 7×7 spatial resolution. Grad-CAM heatmaps are upsampled from 7×7, resulting in coarse localization.

### 8. Single Model Architecture
**Status**: By design for Phase 4
**Description**: Only MobileNetV2 trained. EfficientNetB0 comparison planned for later phase.

## Bugs Found

### 1. Grad-CAM Layer Connection Issue (FIXED)
**Status**: Fixed in Phase 5
**Description**: Grad-CAM layer (Conv_1) was nested inside the MobileNetV2 functional model, causing "Output not connected to inputs" error when creating grad_model.
**Fix**: Split model into base_model and rest_model, compute gradients through base_model then rest_model separately.

### 2. Early Stopping Too Aggressive
**Status**: Open
**Description**: Stage 2 early stopping triggered after 1 epoch (val_loss didn't improve from 2.606). This prevented meaningful fine-tuning.
**Suggested Fix**: Increase early stopping patience, reduce learning rate further, or disable early stopping for fine-tuning stage.

### 3. Learning Rate Schedule
**Status**: Open
**Description**: Stage 2 LR of 1e-5 may be too low for effective fine-tuning. The model barely updates weights.

## Suggested Improvements for Next Phase

1. **Retrain with better hyperparameters**: Increase Stage 2 epochs, adjust LR schedule, increase early stopping patience
2. **Add image quality assessment**: Blur detection, exposure check, leaf detection before prediction
3. **Implement uncertainty quantification**: Monte Carlo dropout, ensemble methods
4. **Train EfficientNetB0 for comparison**: As specified in Phase 4 optional task
4. **Improve near-duplicate detection**: Lower pHash threshold, add perceptual similarity checks
5. **Collect more diverse training data**: Especially for underrepresented classes
6. **Implement test-time augmentation (TTA)**: Average predictions over augmented versions
7. **Add out-of-distribution detection**: Use prediction entropy or Mahalanobis distance

## Test Results Summary (Phase 5 Manual Testing)

| Test Case | Result | Notes |
|-----------|--------|-------|
| Healthy leaf | ⚠️ Low confidence | Model predicts random class |
| Diseased leaf | ⚠️ Low confidence | Model predicts random class |
| Non-leaf image (dog) | ⚠️ Low confidence | Warning shown, wrong class predicted |
| Blurry leaf | ⚠️ Low confidence | Warning shown |
| Dark leaf | ⚠️ Low confidence | Warning shown |
| Invalid file (PDF) | ✅ Pass | Proper error message |
| Oversized file | ✅ Pass | 413 error with message |
| Corrupt image | ✅ Pass | Error message shown |
| Small image (<100px) | ✅ Pass | Error message shown |
| Camera capture | ✅ Pass | Opens camera on mobile |
| Drag & drop | ✅ Pass | Preview works |
| Remove image | ✅ Pass | Form resets |
| New prediction | ✅ Pass | Form resets, scrolls up |
| Low confidence warning | ✅ Pass | Banner at 8% confidence |
| Grad-CAM display | ✅ Pass | Heatmap shown side-by-side |

## Honest Assessment of Non-Leaf Image Behavior

**Test Case**: Photo of a dog uploaded to the system
**Actual Result**: Model predicts "Tomato_Early_blight" with ~8% confidence, shows low-confidence warning, generates Grad-CAM highlighting random fur patterns.

**Interpretation**: This is **correct behavior for a closed-set classifier**. The model has never seen dogs during training and has no "none of the above" class. It must pick one of the 14 training classes. The low confidence and warning are the system working as designed. This is explicitly called out as a known limitation and will be addressed in future stages with OOD detection and image quality checks.

---
*Last updated: October 21, 2026*
*Phase: 5 complete*
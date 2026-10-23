# Test Matrix for XAI-PlantDoc

This document records manual test results for Phase 5.

## Test Cases

| # | Test Case | Input | Expected Behavior | Actual Result | Pass/Fail |
|---|-----------|-------|-------------------|---------------|-----------|
| 1 | Healthy leaf | Photo of healthy tomato leaf | Prediction: Tomato_healthy, confidence > 60%, Grad-CAM shown | | |
| 2 | Diseased leaf | Photo of tomato leaf with Early blight | Prediction: Tomato_Early_blight, confidence > 60%, Grad-CAM highlights lesion areas | | |
| 3 | Non-leaf image (dog) | Photo of a dog | Prediction with low confidence, warning banner shown, Grad-CAM may highlight irrelevant areas | | |
| 4 | Blurry leaf | Out-of-focus leaf photo | Low confidence, warning banner shown, Grad-CAM diffuse | | |
| 5 | Dark leaf | Underexposed leaf photo | Low confidence, warning banner shown | | |
| 6 | Invalid file (PDF) | Upload a PDF file | Error: "Invalid file type. Only JPEG and PNG images are allowed." | | |
| 7 | Oversized file (>5MB) | Upload a 10MB image | Error: "File size exceeds 5 MB limit." | | |
| 8 | Corrupt image | Upload corrupted JPEG | Error: "Cannot decode image" | | |
| 9 | Small image (<100x100) | Upload 50x50 pixel image | Error: "Image dimensions below minimum 100x100 px." | | |
| 10 | Camera capture | Use "Take Photo" on mobile | Opens camera, captures image, predicts | | |
| 11 | Drag & drop | Drag image onto drop zone | Preview appears, predict enabled | | |
| 12 | Remove image | Click X on preview | Form resets, predict disabled | | |
| 13 | New prediction | After result, click "New Prediction" | Form resets, scrolls to upload | | |
| 14 | Low confidence warning | Prediction < 60% | Warning banner appears with confidence % | | |
| 15 | Grad-CAM display | Any valid prediction | Heatmap overlay shown side-by-side with original | | |

## Test Results Summary

- Total tests: 15
- Passed: 
- Failed: 
- Notes:

## Non-Leaf Image Test (Expected Behavior)

**Test Case 3**: Non-leaf image (dog)
- The model is a closed-set classifier trained only on 14 plant disease classes.
- When given a non-leaf image, it will still output one of the 14 classes with some confidence.
- This is **expected behavior** for a closed-set classifier.
- The low-confidence warning should trigger (typically < 60% confidence).
- Grad-CAM may highlight random patterns in the image.
- This limitation is documented in KNOWN_LIMITATIONS.md and will be addressed in later stages with image quality checks and uncertainty handling.
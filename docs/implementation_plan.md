# Implementation Plan - Optic Disc/Cup Segmentation System

**Date:** 2025-12-02
**Version:** 1.0

---

## 1. Implementation Order

The implementation will proceed in the following order to enable incremental testing:

1. **image_processing.py** - Core segmentation logic
2. **main.py** - Batch processing orchestration
3. **Testing with sample image** - Validate on image1.jpg
4. **Parameter tuning** - Adjust based on results

---

## 2. File: image_processing.py

### 2.1 Global Configuration Variables

**Variables to define:**
```
# Preprocessing parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Disc detection parameters
DISC_BRIGHTNESS_PERCENTILE = 95
DISC_MIN_RADIUS = 50
DISC_MAX_RADIUS = 300
DISC_MIN_CIRCULARITY = 0.7

# Cup detection parameters
CUP_BRIGHTNESS_PERCENTILE = 98
CUP_MIN_CDR = 0.1
CUP_MAX_CDR = 0.9
CUP_CONCENTRICITY_TOLERANCE = 0.2

# Post-processing parameters
MORPH_KERNEL_SIZE = 5
MIN_HOLE_SIZE = 100
```

### 2.2 Function: preprocess_image(image)

**Purpose:** Enhance image for better disc/cup detection

**Implementation steps:**
1. Convert RGB to LAB color space
2. Extract L (luminance) channel
3. Apply CLAHE on L channel
4. Apply Gaussian blur for noise reduction
5. Return enhanced grayscale image

**Input:** RGB image (HxWx3 numpy array)
**Output:** Enhanced grayscale image (HxW numpy array)

### 2.3 Function: detect_optic_disc(enhanced_image, original_image)

**Purpose:** Identify optic disc region and return mask

**Implementation steps:**
1. Threshold based on DISC_BRIGHTNESS_PERCENTILE
2. Create binary mask of bright regions
3. Apply morphological operations (closing) to fill gaps
4. Find contours
5. Filter contours by:
   - Size (area within reasonable range)
   - Circularity (> DISC_MIN_CIRCULARITY)
   - Location (not at image edges)
6. Select best candidate (largest valid contour)
7. Optional: Refine with Hough Circle Transform
8. Create binary disc mask
9. Return disc mask and disc center coordinates

**Input:** Enhanced grayscale image, original RGB image
**Output:**
- Binary disc mask (HxW numpy array, 0/255)
- Disc center (x, y)
- Disc radius

### 2.4 Function: detect_optic_cup(enhanced_image, disc_mask, disc_center)

**Purpose:** Identify optic cup region within disc

**Implementation steps:**
1. Apply disc mask to enhanced image (ROI isolation)
2. Threshold based on CUP_BRIGHTNESS_PERCENTILE within disc region
3. Create binary mask of brightest pixels
4. Apply morphological operations (opening, closing)
5. Find contours within disc
6. Select contour closest to disc center
7. Validate:
   - Cup area < Disc area
   - CDR within range (CUP_MIN_CDR to CUP_MAX_CDR)
   - Cup center near disc center
8. Create binary cup mask
9. Return cup mask and cup radius

**Input:** Enhanced image, disc mask, disc center
**Output:**
- Binary cup mask (HxW numpy array, 0/255)
- Cup radius

### 2.5 Function: generate_mask(image_shape, disc_mask, cup_mask)

**Purpose:** Create 3-level output mask

**Implementation steps:**
1. Create blank mask (all 255 - background)
2. Set disc region to 128
3. Set cup region to 0 (overwrites disc)
4. Return final mask

**Input:** Image shape, disc binary mask, cup binary mask
**Output:** 3-level mask (HxW numpy array, values: 0, 128, 255)

### 2.6 Function: postprocess_mask(mask)

**Purpose:** Clean up artifacts in final mask

**Implementation steps:**
1. For disc region (value 128):
   - Fill small holes using morphological closing
   - Smooth boundaries
2. For cup region (value 0):
   - Fill small holes
   - Smooth boundaries
3. Ensure cup is completely within disc (validation)
4. Return cleaned mask

**Input:** 3-level mask
**Output:** Cleaned 3-level mask

### 2.7 Function: process_fundus_image(image)

**Purpose:** Main pipeline function

**Implementation steps:**
1. Call preprocess_image(image)
2. Call detect_optic_disc(enhanced_image, image)
3. If disc not found: raise exception or return None
4. Call detect_optic_cup(enhanced_image, disc_mask, disc_center)
5. If cup not found: raise exception or return None
6. Call generate_mask(image.shape, disc_mask, cup_mask)
7. Call postprocess_mask(mask)
8. Return final mask

**Input:** RGB image (numpy array)
**Output:** 3-level mask (numpy array) or None if failed

**Error Handling:**
- Try-catch blocks for each major step
- Return None on failure with error logging

---

## 3. File: main.py

### 3.1 Global Configuration Variables

**Variables to define:**
```
# Paths (absolute or relative)
INPUT_FOLDER = "/workspace/input_images"
OUTPUT_FOLDER = "/workspace/output_masks"

# Supported formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
```

### 3.2 Function: validate_directories()

**Purpose:** Ensure input/output directories exist and are accessible

**Implementation steps:**
1. Check if INPUT_FOLDER exists, raise error if not
2. Create OUTPUT_FOLDER if it doesn't exist
3. Verify write permissions on OUTPUT_FOLDER

**Input:** None (uses global variables)
**Output:** True if valid, raises exception otherwise

### 3.3 Function: get_image_files()

**Purpose:** Scan input folder for valid image files

**Implementation steps:**
1. Use os.listdir or glob to list files in INPUT_FOLDER
2. Filter by SUPPORTED_FORMATS extensions
3. Return list of full file paths

**Input:** None (uses INPUT_FOLDER global)
**Output:** List of image file paths

### 3.4 Function: process_single_image(image_path)

**Purpose:** Process one image and save result

**Implementation steps:**
1. Load image using cv2.imread or PIL
2. Convert to RGB numpy array if needed
3. Call image_processing.process_fundus_image(image)
4. If result is None: return failure status
5. Generate output filename (same name, .bmp extension)
6. Save mask to OUTPUT_FOLDER using cv2.imwrite
7. Return success/failure status with message

**Input:** Image file path
**Output:** Tuple (success: bool, message: str)

### 3.5 Function: main()

**Purpose:** Main execution orchestration

**Implementation steps:**
1. Print start message
2. Call validate_directories()
3. Call get_image_files() → image_list
4. Initialize counters: total, success, failed
5. Loop through image_list:
   a. Print processing message
   b. Call process_single_image(image_path)
   c. Update counters
   d. Log result
6. Print summary statistics
7. Return exit code (0 if all success, 1 if any failures)

**Input:** None
**Output:** Exit code

### 3.6 Main Execution Block

**Implementation:**
```
if __name__ == "__main__":
    main()
```

---

## 4. Implementation Task Breakdown

### Task 1: Implement image_processing.py structure
- Create file with all global variables
- Add docstrings for each function
- Implement function signatures (empty bodies with pass)

### Task 2: Implement preprocessing functions
- Complete preprocess_image() implementation
- Test on sample image, verify output

### Task 3: Implement disc detection
- Complete detect_optic_disc() implementation
- Test on sample image, verify disc mask

### Task 4: Implement cup detection
- Complete detect_optic_cup() implementation
- Test on sample image, verify cup mask

### Task 5: Implement mask generation and post-processing
- Complete generate_mask() implementation
- Complete postprocess_mask() implementation
- Test full pipeline on sample image

### Task 6: Implement main pipeline function
- Complete process_fundus_image() implementation
- Add error handling
- Test end-to-end on sample image

### Task 7: Implement main.py
- Create file with global variables
- Implement all support functions
- Implement main() function
- Add logging/progress output

### Task 8: Integration testing
- Test on image1.jpg
- Compare output with g0001.bmp format
- Verify output structure

### Task 9: Parameter tuning
- Adjust thresholds based on results
- Test on multiple images
- Optimize for dataset characteristics

### Task 10: Final validation
- Run on full dataset (or subset)
- Review results
- Document any limitations

---

## 5. Implementation Dependencies

**Library imports needed:**

**image_processing.py:**
- import cv2
- import numpy as np
- from typing import Tuple, Optional

**main.py:**
- import cv2
- import numpy as np
- import os
- import glob
- from typing import List, Tuple
- import image_processing

---

## 6. Testing Strategy

**Unit Testing (per function):**
- Test each function independently
- Verify input/output types
- Check edge cases

**Integration Testing:**
- Test complete pipeline on image1.jpg
- Verify output format matches g0001.bmp structure
- Visual inspection of results

**Batch Testing:**
- Test on small subset (5-10 images)
- Check for consistency
- Identify parameter adjustment needs

---

## 7. Files to Create

1. `/workspace/image_processing.py` - NEW FILE
2. `/workspace/main.py` - NEW FILE (minimal entry point)
3. `/workspace/input_images/` - NEW DIRECTORY (if not exists)
4. `/workspace/output_masks/` - NEW DIRECTORY (if not exists)

---

## 8. Risk Mitigation

**Risk 1: Cannot detect disc in some images**
- Mitigation: Log failed images, continue processing others
- Fallback: Manual review of failed cases

**Risk 2: Cup detection too aggressive or conservative**
- Mitigation: Tunable parameters for easy adjustment
- Validation: CDR ratio checks

**Risk 3: Poor results on edge cases**
- Mitigation: Robust error handling
- Documentation: Known limitations

---

## 9. Success Criteria

**Before proceeding to implementation:**
- All functions clearly specified
- Dependencies identified
- Implementation order established
- Testing approach defined

**After implementation:**
- Successfully processes image1.jpg
- Output format matches g0001.bmp structure
- No crashes on valid input
- Clear error messages on failures

---

## End of Implementation Plan

# Retinal Optic Disc/Cup Segmentation System - Technical Design Document

**Date:** 2025-12-02
**Version:** 1.0
**Purpose:** CDR calculation for glaucoma analysis

---

## 1. System Overview

### 1.1 Objective
Process retinal fundus images to generate mask images identifying:
- Optic Cup (pixel value: 0)
- Optic Disc (pixel value: 128)
- Background (pixel value: 255)

### 1.2 Scope
- Batch processing of hundreds of images
- Classical computer vision approach (no deep learning)
- Single execution batch processing

---

## 2. System Architecture

### 2.1 Module Structure

**Module 1: main.py**
- Responsibilities:
  - Configuration management (global variables)
  - File I/O operations
  - Batch processing orchestration
  - Error logging
  - Progress tracking

**Module 2: image_processing.py**
- Responsibilities:
  - Image preprocessing
  - Optic disc segmentation
  - Optic cup segmentation
  - Mask generation
  - Post-processing

### 2.2 Data Flow

```
[Input Folder]
    ↓
[main.py: Read Image Files]
    ↓
[For Each Image]
    ↓
[image_processing.py: process_fundus_image()]
    ↓
    ├─→ [Preprocessing]
    │       ↓
    ├─→ [Disc Detection]
    │       ↓
    ├─→ [Cup Detection]
    │       ↓
    └─→ [Mask Generation]
    ↓
[main.py: Save Mask]
    ↓
[Output Folder]
```

---

## 3. Module Interface Specifications

### 3.1 main.py Interface

**Global Configuration Variables:**
- INPUT_FOLDER: Path to input images directory
- OUTPUT_FOLDER: Path to output masks directory
- SUPPORTED_FORMATS: List of image file extensions

**Main Function Responsibilities:**
- Validate input/output directories
- Scan input folder for valid image files
- Loop through each image
- Call processing pipeline
- Save output with same filename (different extension: .bmp)
- Log success/failure for each image
- Report final statistics

### 3.2 image_processing.py Interface

**Primary Function:**
- Function Name: process_fundus_image
- Input: RGB image array (numpy array, shape: HxWx3)
- Output: Grayscale mask array (numpy array, shape: HxW, dtype: uint8)
- Return Values: 0 (cup), 128 (disc), 255 (background)

**Supporting Functions:**
- preprocess_image: Enhance image quality
- detect_optic_disc: Identify disc region
- detect_optic_cup: Identify cup region within disc
- generate_mask: Create 3-level mask
- postprocess_mask: Clean up artifacts

---

## 4. Algorithm Design

### 4.1 Preprocessing Strategy

**Goal:** Enhance optic disc/cup visibility

**Approach:**
1. Color space conversion
   - RGB → LAB or HSV for better channel separation
   - Extract luminance/brightness channel

2. Contrast enhancement
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Focus on bright regions

3. Noise reduction
   - Gaussian blur for smoothing
   - Preserve edge information

### 4.2 Optic Disc Detection Strategy

**Goal:** Identify the larger circular bright region

**Characteristics to Exploit:**
- Brightest region in fundus image
- Roughly circular shape
- Defined boundaries
- Blood vessels emanate from it

**Approach:**
1. Identify bright regions
   - Threshold on brightness channel
   - Identify top percentile bright pixels

2. Circular region detection
   - Hough Circle Transform
   - OR morphological operations (closing, filling)

3. Boundary refinement
   - Edge detection (Canny)
   - Contour analysis
   - Ellipse/circle fitting

4. Validation
   - Size constraints (reasonable diameter range)
   - Shape circularity metric
   - Location constraints (typically not at edges)

### 4.3 Optic Cup Detection Strategy

**Goal:** Identify the brighter central depression within disc

**Characteristics to Exploit:**
- Brightest region within optic disc
- Smaller than disc
- Roughly concentric with disc
- More uniform color (less vascular)

**Approach:**
1. Region of Interest isolation
   - Mask to disc region only
   - Focus processing on disc area

2. Intensity-based segmentation
   - Higher threshold than disc
   - Identify brightest pixels within disc

3. Pallor detection
   - Color uniformity analysis
   - Yellow/pale region identification

4. Shape refinement
   - Morphological operations
   - Ensure cup is within disc boundaries
   - Smooth irregular boundaries

5. Validation
   - Cup diameter < Disc diameter
   - Concentric with disc
   - CDR reasonableness check (0.1 - 0.9)

### 4.4 Mask Generation Strategy

**Goal:** Create 3-level output mask

**Approach:**
1. Initialize blank mask (all 255)
2. Fill disc region with value 128
3. Fill cup region with value 0
4. Ensure proper layering (cup overwrites disc)

### 4.5 Post-Processing Strategy

**Goal:** Clean up artifacts and ensure quality

**Approach:**
1. Morphological operations
   - Remove small holes in regions
   - Smooth boundaries
   - Fill gaps

2. Validation checks
   - Cup completely within disc
   - Single connected component for each region
   - Reasonable size ratios

---

## 5. Technical Specifications

### 5.1 Image Specifications

**Input:**
- Format: JPG (from dataset)
- Color: RGB
- Resolution: Variable (high resolution)

**Output:**
- Format: BMP
- Color: Grayscale (8-bit)
- Values: 0, 128, 255 only
- Resolution: Same as input

### 5.2 Performance Considerations

**Processing Time:**
- Target: < 5 seconds per image (acceptable for batch)
- Classical CV is CPU-bound
- No GPU required

**Memory:**
- Load one image at a time
- Clear intermediate results
- Efficient for hundreds of images

### 5.3 Error Handling

**Potential Issues:**
1. Unable to detect disc
2. Unable to detect cup
3. Invalid CDR ratio
4. Image quality issues

**Handling Strategy:**
- Log failures with image filename
- Skip failed images, continue processing
- Optionally save partial results for review
- Generate summary report at end

---

## 6. Quality Metrics

### 6.1 Validation Checks

**Geometric Validation:**
- Cup area < Disc area
- CDR between 0.1 and 0.9
- Both regions roughly circular (circularity > threshold)
- Cup center close to disc center

**Visual Validation:**
- Manual review of sample outputs
- Compare with ground truth if available

---

## 7. Configuration Parameters

### 7.1 Tunable Parameters (as global variables in image_processing.py)

**Preprocessing:**
- CLAHE clip limit
- Gaussian blur kernel size

**Disc Detection:**
- Brightness threshold percentile
- Circle detection parameters (min/max radius)
- Minimum circularity

**Cup Detection:**
- Intensity threshold (relative to disc)
- Size ratio constraints (cup/disc ratio range)
- Concentricity tolerance

**Post-Processing:**
- Morphological kernel sizes
- Hole filling threshold

---

## 8. Project Structure

```
/workspace/
├── main.py                          # Entry point, batch processing
├── image_processing.py              # Segmentation pipeline
├── docs/
│   └── technical_design.md          # This document
├── input_images/                    # Default input folder (configurable)
└── output_masks/                    # Default output folder (configurable)
```

---

## 9. Implementation Notes

### 9.1 Libraries Required
- OpenCV (cv2): Image processing operations
- NumPy: Array operations
- Pillow (PIL): Image I/O operations
- os/glob: File operations

### 9.2 Development Approach
- Start with simple thresholding approach
- Iteratively refine based on results
- Test on sample images before batch processing
- Adjust parameters based on dataset characteristics

---

## 10. Future Enhancements (Out of Scope)

- Deep learning based segmentation
- Real-time processing
- GUI interface
- Automatic parameter tuning
- Multi-threading for parallel processing
- Quality confidence scoring

---

## End of Design Document

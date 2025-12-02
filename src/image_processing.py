"""
Retinal Fundus Image Processing Module
Optic Disc and Cup Segmentation Pipeline

This module provides functions to segment optic disc and cup regions
from retinal fundus images for CDR (Cup-to-Disc Ratio) calculation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Preprocessing parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Disc detection parameters
DISC_BRIGHTNESS_PERCENTILE = 65  # Low value to avoid bright background outside retina
DISC_MIN_RADIUS = 50
DISC_MAX_RADIUS = 300
DISC_MIN_CIRCULARITY = 0.6  # Lowered from 0.7 for more tolerance

# Cup detection parameters
CUP_BRIGHTNESS_PERCENTILE = 98
CUP_MIN_CDR = 0.1
CUP_MAX_CDR = 0.9
CUP_CONCENTRICITY_TOLERANCE = 0.2

# Post-processing parameters
MORPH_KERNEL_SIZE = 5
MIN_HOLE_SIZE = 100

# Debug configuration
DEBUG_MODE = True  # Master switch - set to True to enable debug output
DEBUG_OUTPUT_FOLDER = "./debug_images"

# Granular stage control - enable/disable individual stages
DEBUG_STAGES = {
    'preprocessing': True,      # Stage 1: Green channel, CLAHE, blur
    'disc_detection': False,     # Stage 2: Disc detection steps
    'cup_detection': False,      # Stage 3: Cup detection steps
    'mask_generation': False,    # Stage 4: Combining masks
    'postprocessing': False,     # Stage 5: Final cleanup
    'overlays': False,           # Overlay visualizations on original
}

# Global prefix for current image being processed
_DEBUG_PREFIX = ""

# ============================================================================
# DEBUG HELPER FUNCTIONS
# ============================================================================

import os

def is_debug_stage_enabled(stage_name: str) -> bool:
    """
    Check if debug output is enabled for a specific stage.

    Args:
        stage_name: Name of the stage to check

    Returns:
        True if both DEBUG_MODE and the specific stage are enabled
    """
    return DEBUG_MODE and DEBUG_STAGES.get(stage_name, False)


def save_debug_image(image: np.ndarray, filename: str, stage_name: str):
    """
    Save debug image if the stage is enabled.

    Args:
        image: Image to save (numpy array)
        filename: Filename for the debug image
        stage_name: Stage name to check if enabled
    """
    if not is_debug_stage_enabled(stage_name):
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(DEBUG_OUTPUT_FOLDER):
        os.makedirs(DEBUG_OUTPUT_FOLDER)

    # Build full path
    if _DEBUG_PREFIX:
        full_filename = f"{_DEBUG_PREFIX}_{filename}"
    else:
        full_filename = filename

    output_path = os.path.join(DEBUG_OUTPUT_FOLDER, full_filename)

    # Save image
    cv2.imwrite(output_path, image)


def set_debug_prefix(prefix: str):
    """
    Set the prefix for debug output filenames.

    Args:
        prefix: Prefix string (usually image filename without extension)
    """
    global _DEBUG_PREFIX
    _DEBUG_PREFIX = prefix


def enable_all_debug_stages():
    """Helper function to enable all debug stages at once."""
    global DEBUG_STAGES
    DEBUG_STAGES = {k: True for k in DEBUG_STAGES}


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better disc/cup detection using CLAHE and color space conversion.

    Args:
        image: Input RGB image (HxWx3 numpy array)

    Returns:
        Enhanced grayscale image (HxW numpy array)
    """
    # Extract green channel (best for fundus images - optic disc is brightest here)
    green_channel = image[:, :, 1]
    save_debug_image(green_channel, "stage1a_green_channel.jpg", 'preprocessing')

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(green_channel)
    save_debug_image(enhanced, "stage1b_clahe_enhanced.jpg", 'preprocessing')

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, GAUSSIAN_BLUR_KERNEL, 0)
    save_debug_image(blurred, "stage1c_final_preprocessed.jpg", 'preprocessing')

    return blurred


# ============================================================================
# DISC DETECTION FUNCTIONS
# ============================================================================

def detect_optic_disc(enhanced_image: np.ndarray,
                      original_image: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                             Optional[Tuple[int, int]],
                                                             Optional[int]]:
    """
    Identify optic disc region and return mask with center and radius.

    Args:
        enhanced_image: Enhanced grayscale image from preprocessing
        original_image: Original RGB image for reference

    Returns:
        Tuple containing:
        - Binary disc mask (HxW numpy array, 0/255) or None if not found
        - Disc center coordinates (x, y) or None
        - Disc radius (int) or None
    """
    h, w = enhanced_image.shape

    # Step 1: First, create retinal area mask (exclude bright background)
    # Pixels with value 255 or very close to it are background
    _, background_mask = cv2.threshold(enhanced_image, 250, 255, cv2.THRESH_BINARY)
    save_debug_image(background_mask, "stage2a_background_mask.jpg", 'disc_detection')

    retinal_mask = cv2.bitwise_not(background_mask)

    # Apply morphology to clean up retinal mask
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    retinal_mask = cv2.morphologyEx(retinal_mask, cv2.MORPH_CLOSE, kernel_large)
    retinal_mask = cv2.morphologyEx(retinal_mask, cv2.MORPH_OPEN, kernel_large)
    save_debug_image(retinal_mask, "stage2b_retinal_mask.jpg", 'disc_detection')

    # Step 2: Within retinal area, find bright regions (optic disc)
    # Mask the enhanced image to retinal area only
    retinal_only = cv2.bitwise_and(enhanced_image, enhanced_image, mask=retinal_mask)
    save_debug_image(retinal_only, "stage2c_retinal_only.jpg", 'disc_detection')

    # Get pixels within retinal area
    retinal_pixels = enhanced_image[retinal_mask > 0]
    if len(retinal_pixels) == 0:
        return None, None, None

    # Use high percentile within retinal area to find optic disc
    threshold_value = np.percentile(retinal_pixels, 90)

    _, binary = cv2.threshold(retinal_only, threshold_value, 255, cv2.THRESH_BINARY)
    save_debug_image(binary, "stage2d_disc_threshold.jpg", 'disc_detection')

    # Step 2: Apply morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    save_debug_image(closed, "stage2e_disc_morphology.jpg", 'disc_detection')

    # Step 3: Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # Step 4: Filter and select best candidate
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Calculate equivalent radius
        if area == 0:
            continue
        radius = int(np.sqrt(area / np.pi))

        # Filter by radius range
        if radius < DISC_MIN_RADIUS or radius > DISC_MAX_RADIUS:
            continue

        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Filter by circularity
        if circularity < DISC_MIN_CIRCULARITY:
            continue

        # Get center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Filter by location (not too close to edges)
        margin = 50
        if cx < margin or cx > w - margin or cy < margin or cy > h - margin:
            continue

        valid_contours.append({
            'contour': contour,
            'area': area,
            'center': (cx, cy),
            'radius': radius,
            'circularity': circularity
        })

    if not valid_contours:
        return None, None, None

    # Step 5: Select largest valid contour as disc
    best_disc = max(valid_contours, key=lambda x: x['area'])

    # Step 6: Create disc mask
    disc_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(disc_mask, [best_disc['contour']], -1, 255, -1)
    save_debug_image(disc_mask, "stage2f_disc_mask_final.jpg", 'disc_detection')

    return disc_mask, best_disc['center'], best_disc['radius']


# ============================================================================
# CUP DETECTION FUNCTIONS
# ============================================================================

def detect_optic_cup(enhanced_image: np.ndarray,
                     disc_mask: np.ndarray,
                     disc_center: Tuple[int, int],
                     disc_radius: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Identify optic cup region within disc and return mask.

    Args:
        enhanced_image: Enhanced grayscale image
        disc_mask: Binary mask of disc region
        disc_center: Center coordinates of disc (x, y)
        disc_radius: Radius of disc

    Returns:
        Tuple containing:
        - Binary cup mask (HxW numpy array, 0/255) or None if not found
        - Cup radius (int) or None
    """
    h, w = enhanced_image.shape

    # Step 1: Isolate disc region (ROI)
    roi_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=disc_mask)
    save_debug_image(roi_image, "stage3a_disc_roi.jpg", 'cup_detection')

    # Step 2: Threshold based on cup brightness percentile within disc
    disc_pixels = enhanced_image[disc_mask > 0]
    if len(disc_pixels) == 0:
        return None, None

    threshold_value = np.percentile(disc_pixels, CUP_BRIGHTNESS_PERCENTILE)
    _, binary = cv2.threshold(roi_image, threshold_value, 255, cv2.THRESH_BINARY)
    save_debug_image(binary, "stage3b_cup_threshold.jpg", 'cup_detection')

    # Step 3: Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    save_debug_image(closed, "stage3c_cup_morphology.jpg", 'cup_detection')

    # Step 4: Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Step 5: Select contour closest to disc center
    cx_disc, cy_disc = disc_center
    best_cup = None
    min_distance = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0:
            continue

        # Calculate cup radius
        cup_radius = int(np.sqrt(area / np.pi))

        # Get center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate distance to disc center
        distance = np.sqrt((cx - cx_disc)**2 + (cy - cy_disc)**2)

        # Calculate CDR
        cdr = cup_radius / disc_radius

        # Validate CDR ratio
        if cdr < CUP_MIN_CDR or cdr > CUP_MAX_CDR:
            continue

        # Validate concentricity (cup center close to disc center)
        concentricity_ratio = distance / disc_radius
        if concentricity_ratio > CUP_CONCENTRICITY_TOLERANCE:
            continue

        # Select closest to center
        if distance < min_distance:
            min_distance = distance
            best_cup = {
                'contour': contour,
                'radius': cup_radius,
                'center': (cx, cy)
            }

    if best_cup is None:
        return None, None

    # Step 6: Create cup mask
    cup_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cup_mask, [best_cup['contour']], -1, 255, -1)

    # Ensure cup is within disc (intersection)
    cup_mask = cv2.bitwise_and(cup_mask, cup_mask, mask=disc_mask)
    save_debug_image(cup_mask, "stage3d_cup_mask_final.jpg", 'cup_detection')

    return cup_mask, best_cup['radius']


# ============================================================================
# MASK GENERATION FUNCTIONS
# ============================================================================

def generate_mask(image_shape: Tuple[int, int],
                  disc_mask: np.ndarray,
                  cup_mask: np.ndarray) -> np.ndarray:
    """
    Create 3-level output mask with background=255, disc=128, cup=0.

    Args:
        image_shape: Shape of output mask (height, width)
        disc_mask: Binary mask of disc region
        cup_mask: Binary mask of cup region

    Returns:
        3-level mask (HxW numpy array, values: 0, 128, 255)
    """
    # Initialize with background (255)
    mask = np.full(image_shape, 255, dtype=np.uint8)

    # Set disc region to 128
    mask[disc_mask > 0] = 128

    # Set cup region to 0 (overwrites disc)
    mask[cup_mask > 0] = 0

    save_debug_image(mask, "stage4_combined_mask.jpg", 'mask_generation')

    return mask


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up artifacts in final mask using morphological operations.

    Args:
        mask: 3-level mask to be cleaned

    Returns:
        Cleaned 3-level mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    # Process disc region (value 128)
    disc_region = (mask == 128).astype(np.uint8) * 255
    disc_cleaned = cv2.morphologyEx(disc_region, cv2.MORPH_CLOSE, kernel)
    save_debug_image(disc_cleaned, "stage5a_disc_cleaned.jpg", 'postprocessing')

    # Process cup region (value 0)
    cup_region = (mask == 0).astype(np.uint8) * 255
    cup_cleaned = cv2.morphologyEx(cup_region, cv2.MORPH_CLOSE, kernel)
    save_debug_image(cup_cleaned, "stage5b_cup_cleaned.jpg", 'postprocessing')

    # Reconstruct mask
    cleaned_mask = np.full(mask.shape, 255, dtype=np.uint8)
    cleaned_mask[disc_cleaned > 0] = 128
    cleaned_mask[cup_cleaned > 0] = 0

    save_debug_image(cleaned_mask, "stage5c_final_mask.jpg", 'postprocessing')

    return cleaned_mask


def generate_debug_overlays(original_image: np.ndarray,
                             disc_mask: Optional[np.ndarray],
                             cup_mask: Optional[np.ndarray],
                             final_mask: Optional[np.ndarray]):
    """
    Generate overlay visualizations showing disc and cup boundaries on original image.

    Args:
        original_image: Original RGB fundus image
        disc_mask: Binary disc mask (can be None)
        cup_mask: Binary cup mask (can be None)
        final_mask: Final 3-level mask (can be None)
    """
    if not is_debug_stage_enabled('overlays'):
        return

    # Convert original to BGR for OpenCV drawing
    overlay = original_image.copy()
    if overlay.shape[2] == 3:  # RGB
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    else:
        overlay_bgr = overlay.copy()

    # Overlay with disc boundary only
    if disc_mask is not None:
        disc_overlay = overlay_bgr.copy()
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disc_overlay, disc_contours, -1, (0, 255, 0), 3)  # Green for disc
        save_debug_image(disc_overlay, "overlay_disc.jpg", 'overlays')

    # Overlay with cup boundary only
    if cup_mask is not None:
        cup_overlay = overlay_bgr.copy()
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cup_overlay, cup_contours, -1, (0, 0, 255), 3)  # Red for cup
        save_debug_image(cup_overlay, "overlay_cup.jpg", 'overlays')

    # Overlay with both disc and cup boundaries
    if disc_mask is not None and cup_mask is not None:
        both_overlay = overlay_bgr.copy()
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(both_overlay, disc_contours, -1, (0, 255, 0), 3)  # Green for disc
        cv2.drawContours(both_overlay, cup_contours, -1, (0, 0, 255), 3)   # Red for cup
        save_debug_image(both_overlay, "overlay_both.jpg", 'overlays')

    # Colored mask overlay (semi-transparent)
    if final_mask is not None:
        colored_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        # Disc = green (128 value)
        colored_mask[final_mask == 128] = [0, 255, 0]
        # Cup = red (0 value)
        colored_mask[final_mask == 0] = [0, 0, 255]

        # Blend with original
        mask_overlay = cv2.addWeighted(overlay_bgr, 0.6, colored_mask, 0.4, 0)
        save_debug_image(mask_overlay, "overlay_mask_colored.jpg", 'overlays')


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def process_fundus_image(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Main pipeline function to process fundus image and generate segmentation mask.

    This function orchestrates the complete segmentation pipeline:
    1. Preprocess image
    2. Detect optic disc
    3. Detect optic cup
    4. Generate mask
    5. Post-process mask

    Args:
        image: Input RGB fundus image (HxWx3 numpy array)

    Returns:
        3-level segmentation mask (HxW numpy array with values 0, 128, 255)
        or None if segmentation fails
    """
    try:
        # Step 1: Preprocess image
        enhanced_image = preprocess_image(image)

        # Step 2: Detect optic disc
        disc_mask, disc_center, disc_radius = detect_optic_disc(enhanced_image, image)

        if disc_mask is None:
            print("Error: Could not detect optic disc")
            return None

        # Step 3: Detect optic cup
        cup_mask, cup_radius = detect_optic_cup(enhanced_image, disc_mask,
                                                 disc_center, disc_radius)

        if cup_mask is None:
            print("Error: Could not detect optic cup")
            return None

        # Step 4: Generate mask
        mask = generate_mask(image.shape[:2], disc_mask, cup_mask)

        # Step 5: Post-process mask
        final_mask = postprocess_mask(mask)

        # Step 6: Generate debug overlays if enabled
        generate_debug_overlays(image, disc_mask, cup_mask, final_mask)

        return final_mask

    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        return None

"""
Retinal Fundus Image Batch Processing
Main entry point for optic disc/cup segmentation

This script processes all fundus images in the input folder and generates
segmentation masks in the output folder.
"""

import cv2
import numpy as np
import os
import glob
from typing import List, Tuple
from src import image_processing

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Input/Output folder paths (modify these as needed)
INPUT_FOLDER = "./input_images"
OUTPUT_FOLDER = "./output_masks"

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_directories() -> bool:
    """
    Ensure input/output directories exist and are accessible.

    Returns:
        True if valid, raises exception otherwise
    """
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    if not os.path.isdir(INPUT_FOLDER):
        raise NotADirectoryError(f"Input path is not a directory: {INPUT_FOLDER}")

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Verify write permissions on output folder
    if not os.access(OUTPUT_FOLDER, os.W_OK):
        raise PermissionError(f"No write permission for output folder: {OUTPUT_FOLDER}")

    return True


def get_image_files() -> List[str]:
    """
    Scan input folder for valid image files.

    Returns:
        List of image file paths
    """
    image_files = []

    for ext in SUPPORTED_FORMATS:
        # Search for files with each supported extension (case-insensitive)
        pattern = os.path.join(INPUT_FOLDER, f"*{ext}")
        image_files.extend(glob.glob(pattern))

        # Also search for uppercase extensions
        pattern_upper = os.path.join(INPUT_FOLDER, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern_upper))

    return sorted(image_files)


def process_single_image(image_path: str) -> Tuple[bool, str]:
    """
    Process one image and save result.

    Args:
        image_path: Path to input image

    Returns:
        Tuple (success: bool, message: str)
    """
    try:
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            return False, "Failed to load image"

        # Convert BGR to RGB (OpenCV loads as BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        mask = image_processing.process_fundus_image(image_rgb)

        if mask is None:
            return False, "Segmentation failed"

        # Generate output filename: <input_filename>.bmp
        input_filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(input_filename)[0]
        output_filename = f"{filename_without_ext}.bmp"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Save mask as BMP
        success = cv2.imwrite(output_path, mask)

        if not success:
            return False, "Failed to save output image"

        return True, output_path

    except Exception as e:
        return False, f"Exception: {str(e)}"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main execution orchestration for batch processing.

    Returns:
        Exit code (0 if all success, 1 if any failures)
    """
    print("=" * 70)
    print("Retinal Fundus Image Processing - Optic Disc/Cup Segmentation")
    print("=" * 70)
    print()

    # Validate directories
    try:
        validate_directories()
        print(f"Input folder:  {INPUT_FOLDER}")
        print(f"Output folder: {OUTPUT_FOLDER}")
        print()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    # Get list of images
    image_list = get_image_files()

    if not image_list:
        print("No images found in input folder.")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return 1

    print(f"Found {len(image_list)} images to process.")
    print()

    # Initialize counters
    total = len(image_list)
    success_count = 0
    failed_count = 0
    failed_images = []

    # Process each image
    for idx, image_path in enumerate(image_list, 1):
        filename = os.path.basename(image_path)
        print(f"[{idx}/{total}] Processing: {filename}... ", end='', flush=True)

        success, message = process_single_image(image_path)

        if success:
            success_count += 1
            output_filename = os.path.basename(message)
            print(f"✓ Success → {output_filename}")
        else:
            failed_count += 1
            failed_images.append((filename, message))
            print(f"✗ Failed: {message}")

    # Print summary
    print()
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total images:      {total}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed:            {failed_count}")
    print()

    if failed_images:
        print("Failed images:")
        for filename, reason in failed_images:
            print(f"  - {filename}: {reason}")
        print()

    # Return appropriate exit code
    return 0 if failed_count == 0 else 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

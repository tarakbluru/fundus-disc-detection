"""
Debug script to visualize cup detection process
"""

import sys
import os
# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src import image_processing

# Load image
image = cv2.imread('/workspace/input_images/image1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Image shape: {image_rgb.shape}")

# Step 1: Preprocess
enhanced = image_processing.preprocess_image(image_rgb)

# Step 2: Detect disc
disc_mask, disc_center, disc_radius = image_processing.detect_optic_disc(enhanced, image_rgb)

if disc_mask is None:
    print("ERROR: Disc not detected!")
    exit(1)

print(f"\nDisc detected:")
print(f"  Center: {disc_center}")
print(f"  Radius: {disc_radius}")

# Save disc mask
cv2.imwrite('/workspace/debug_disc_mask.jpg', disc_mask)

# Step 3: Try cup detection
roi_image = cv2.bitwise_and(enhanced, enhanced, mask=disc_mask)
cv2.imwrite('/workspace/debug_roi.jpg', roi_image)

disc_pixels = enhanced[disc_mask > 0]
print(f"\nDisc region pixels: {len(disc_pixels)}")
print(f"  Min: {disc_pixels.min()}, Max: {disc_pixels.max()}, Mean: {disc_pixels.mean():.2f}")
print(f"  Percentiles in disc region:")
for p in [70, 80, 90, 95, 98, 99]:
    val = np.percentile(disc_pixels, p)
    print(f"    {p}th: {val:.1f}")

threshold_value = np.percentile(disc_pixels, image_processing.CUP_BRIGHTNESS_PERCENTILE)
print(f"\nCup brightness percentile: {image_processing.CUP_BRIGHTNESS_PERCENTILE}")
print(f"Threshold value: {threshold_value}")

_, binary = cv2.threshold(roi_image, threshold_value, 255, cv2.THRESH_BINARY)
print(f"Binary mask - white pixels: {np.sum(binary > 0)}")
cv2.imwrite('/workspace/debug_cup_binary.jpg', binary)

# Morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('/workspace/debug_cup_closed.jpg', closed)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nNumber of cup contours found: {len(contours)}")

if contours:
    cx_disc, cy_disc = disc_center
    print("\nCup contour analysis:")
    for i, contour in enumerate(contours[:10]):
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        cup_radius = int(np.sqrt(area / np.pi))
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        distance = np.sqrt((cx - cx_disc)**2 + (cy - cy_disc)**2)
        cdr = cup_radius / disc_radius
        concentricity_ratio = distance / disc_radius

        print(f"  Contour {i}: area={area:.0f}, cup_radius={cup_radius}, cdr={cdr:.3f}")
        print(f"             center=({cx},{cy}), distance={distance:.1f}, conc_ratio={concentricity_ratio:.3f}")

        # Check filters
        if cdr < image_processing.CUP_MIN_CDR:
            print(f"    → Filtered: CDR {cdr:.3f} < min {image_processing.CUP_MIN_CDR}")
        elif cdr > image_processing.CUP_MAX_CDR:
            print(f"    → Filtered: CDR {cdr:.3f} > max {image_processing.CUP_MAX_CDR}")
        elif concentricity_ratio > image_processing.CUP_CONCENTRICITY_TOLERANCE:
            print(f"    → Filtered: Concentricity {concentricity_ratio:.3f} > max {image_processing.CUP_CONCENTRICITY_TOLERANCE}")
        else:
            print(f"    → VALID")

print("\nDebug images saved:")
print("  - debug_disc_mask.jpg")
print("  - debug_roi.jpg")
print("  - debug_cup_binary.jpg")
print("  - debug_cup_closed.jpg")

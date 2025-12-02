"""
Debug script to visualize disc detection process
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
print(f"Enhanced image shape: {enhanced.shape}")
print(f"Enhanced image dtype: {enhanced.dtype}")
print(f"Enhanced image range: [{enhanced.min()}, {enhanced.max()}]")

# Save enhanced image for inspection
cv2.imwrite('/workspace/debug_enhanced.jpg', enhanced)

# Step 2: Check threshold
threshold_value = np.percentile(enhanced, image_processing.DISC_BRIGHTNESS_PERCENTILE)
print(f"\nDisc brightness percentile: {image_processing.DISC_BRIGHTNESS_PERCENTILE}")
print(f"Threshold value: {threshold_value}")

_, binary = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)
print(f"Binary mask - white pixels: {np.sum(binary > 0)}")

cv2.imwrite('/workspace/debug_binary.jpg', binary)

# Step 3: Morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('/workspace/debug_closed.jpg', closed)

# Step 4: Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nNumber of contours found: {len(contours)}")

if contours:
    print("\nContour analysis:")
    for i, contour in enumerate(contours[:10]):  # Show first 10
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        radius = int(np.sqrt(area / np.pi))
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        print(f"  Contour {i}: area={area:.0f}, radius={radius}, circularity={circularity:.3f}, center=({cx},{cy})")

        # Check filters
        if radius < image_processing.DISC_MIN_RADIUS:
            print(f"    → Filtered: radius {radius} < min {image_processing.DISC_MIN_RADIUS}")
        elif radius > image_processing.DISC_MAX_RADIUS:
            print(f"    → Filtered: radius {radius} > max {image_processing.DISC_MAX_RADIUS}")
        elif circularity < image_processing.DISC_MIN_CIRCULARITY:
            print(f"    → Filtered: circularity {circularity:.3f} < min {image_processing.DISC_MIN_CIRCULARITY}")
        else:
            print(f"    → VALID")

print("\nDebug images saved:")
print("  - debug_enhanced.jpg")
print("  - debug_binary.jpg")
print("  - debug_closed.jpg")

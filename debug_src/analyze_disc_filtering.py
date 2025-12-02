"""Analyze why disc detection is failing - check filters"""
import cv2
import numpy as np

# Load morphology image
img = cv2.imread('debug_output/test2_all_stage2e_disc_morphology.jpg', 0)
h, w = img.shape

# Parameters from image_processing.py
DISC_MIN_RADIUS = 50
DISC_MAX_RADIUS = 300
DISC_MIN_CIRCULARITY = 0.6

# Find contours
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}\n")

# Analyze largest 10 contours
contour_data = []
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area == 0:
        continue
    contour_data.append((i, contour, area))

# Sort by area
contour_data.sort(key=lambda x: x[2], reverse=True)

print("Top 10 contours analysis:")
print("=" * 80)

for idx, (original_idx, contour, area) in enumerate(contour_data[:10]):
    radius = int(np.sqrt(area / np.pi))

    # Circularity
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Center
    M = cv2.moments(contour)
    if M["m00"] == 0:
        cx, cy = 0, 0
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    # Location check
    margin = 50
    location_ok = not (cx < margin or cx > w - margin or cy < margin or cy > h - margin)

    print(f"\nContour {idx + 1} (index {original_idx}):")
    print(f"  Area: {area:.1f}")
    print(f"  Radius: {radius} px")
    print(f"  Circularity: {circularity:.3f}")
    print(f"  Center: ({cx}, {cy})")
    print(f"  Image size: ({w}, {h})")

    # Filter checks
    passed_all = True
    if radius < DISC_MIN_RADIUS:
        print(f"  ✗ REJECTED: Radius {radius} < min {DISC_MIN_RADIUS}")
        passed_all = False
    elif radius > DISC_MAX_RADIUS:
        print(f"  ✗ REJECTED: Radius {radius} > max {DISC_MAX_RADIUS}")
        passed_all = False
    else:
        print(f"  ✓ Radius OK: {DISC_MIN_RADIUS} <= {radius} <= {DISC_MAX_RADIUS}")

    if circularity < DISC_MIN_CIRCULARITY:
        print(f"  ✗ REJECTED: Circularity {circularity:.3f} < min {DISC_MIN_CIRCULARITY}")
        passed_all = False
    else:
        print(f"  ✓ Circularity OK: {circularity:.3f} >= {DISC_MIN_CIRCULARITY}")

    if not location_ok:
        print(f"  ✗ REJECTED: Too close to edge (margin={margin})")
        passed_all = False
    else:
        print(f"  ✓ Location OK: Center within margins")

    if passed_all:
        print(f"  ✓✓ PASSES ALL FILTERS - Would be selected as disc")

print("\n" + "=" * 80)
print("Analysis complete")

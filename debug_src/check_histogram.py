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

# Get green channel
green = image_rgb[:, :, 1]
print(f"Green channel stats:")
print(f"  Min: {green.min()}, Max: {green.max()}, Mean: {green.mean():.2f}")
print(f"  Percentiles:")
for p in [50, 70, 80, 85, 90, 95, 98, 99]:
    val = np.percentile(green, p)
    print(f"    {p}th: {val:.1f}")

# Get enhanced
enhanced = image_processing.preprocess_image(image_rgb)
print(f"\nEnhanced image stats:")
print(f"  Min: {enhanced.min()}, Max: {enhanced.max()}, Mean: {enhanced.mean():.2f}")
print(f"  Percentiles:")
for p in [50, 70, 80, 85, 90, 95, 98, 99]:
    val = np.percentile(enhanced, p)
    print(f"    {p}th: {val:.1f}")

# Count pixels at 255
count_255 = np.sum(enhanced == 255)
total_pixels = enhanced.size
pct_255 = (count_255 / total_pixels) * 100
print(f"\nPixels with value 255: {count_255}/{total_pixels} ({pct_255:.2f}%)")

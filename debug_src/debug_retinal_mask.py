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
enhanced = image_processing.preprocess_image(image_rgb)

# Create retinal mask
_, background_mask = cv2.threshold(enhanced, 250, 255, cv2.THRESH_BINARY)
retinal_mask = cv2.bitwise_not(background_mask)
cv2.imwrite('/workspace/debug_retinal_mask_raw.jpg', retinal_mask)

kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
retinal_mask_closed = cv2.morphologyEx(retinal_mask, cv2.MORPH_CLOSE, kernel_large)
retinal_mask_final = cv2.morphologyEx(retinal_mask_closed, cv2.MORPH_OPEN, kernel_large)
cv2.imwrite('/workspace/debug_retinal_mask_final.jpg', retinal_mask_final)

retinal_pixels = enhanced[retinal_mask_final > 0]
print(f"Retinal pixels: {len(retinal_pixels)}")
if len(retinal_pixels) > 0:
    print(f"  Min: {retinal_pixels.min()}, Max: {retinal_pixels.max()}, Mean: {retinal_pixels.mean():.2f}")
    print(f"  90th percentile: {np.percentile(retinal_pixels, 90)}")

    retinal_only = cv2.bitwise_and(enhanced, enhanced, mask=retinal_mask_final)
    cv2.imwrite('/workspace/debug_retinal_only.jpg', retinal_only)

    threshold_value = np.percentile(retinal_pixels, 90)
    _, binary = cv2.threshold(retinal_only, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite('/workspace/debug_disc_binary.jpg', binary)
    print(f"Binary white pixels: {np.sum(binary > 0)}")

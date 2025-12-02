import cv2
import numpy as np
from PIL import Image

# Load the image
img = cv2.imread('/workspace/g0001.bmp', cv2.IMREAD_UNCHANGED)
print(f"Shape: {img.shape}")
print(f"Data type: {img.dtype}")
print(f"Unique values: {np.unique(img)}")
print(f"Min: {img.min()}, Max: {img.max()}")

# Convert to RGB for PIL
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

# Save as PNG for viewing
pil_img.save('/workspace/g0001_view.png')
print("\nSaved as g0001_view.png")

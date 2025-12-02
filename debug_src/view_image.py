from PIL import Image
import sys

# Load and display image info
img = Image.open('/workspace/g0001.bmp')
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")

# Convert to smaller format for viewing
img.save('/workspace/g0001_preview.png')
print("Saved preview as g0001_preview.png")

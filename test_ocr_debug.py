import cv2
import easyocr
import numpy as np

# Load image
img = cv2.imread('81suTTBxfuL.jpg')
print(f"Image shape: {img.shape}")

# Initialize reader
reader = easyocr.Reader(['en'], gpu=False, verbose=True)

# Try reading
print("Running OCR...")
results = reader.readtext(img)

print(f"\nTotal results: {len(results)}")
for i, r in enumerate(results[:10]):
    print(f"{i}: {r}")


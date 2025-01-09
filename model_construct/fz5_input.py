import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define file paths
input_file_path = "./data/images/01_green.png"
output_file_path = "./fz5/01_green_128x128.bin"

# Define cropping margin
CROP_MARGIN = 0

# Function to crop the image
def crop_image(image, margin):
    """
    Crops 'margin' pixels from each side of the image.
    If the image is too small to crop this amount, it returns the original image.
    """
    h, w = image.shape[:2]
    
    # Check if there's enough room to crop
    if margin * 2 >= w or margin * 2 >= h:
        print(f"Warning: cannot crop {margin}px from each side of a {w}x{h} image. Returning original.")
        return image
    
    return image[margin:h-margin, margin:w-margin]

# Step 1: Load the input image
image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Input file not found: {input_file_path}")

# Step 2: Crop the image
cropped_image = crop_image(image, CROP_MARGIN)

# Step 3: Resize to 128x128
resized_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_AREA)

# Step 4: Flatten the image and save as binary
resized_image.flatten().tofile(output_file_path)

# Visualize the original, cropped, and resized images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[1].imshow(cropped_image, cmap="gray")
axes[1].set_title("Cropped Image")
axes[2].imshow(resized_image, cmap="gray")
axes[2].set_title("Resized to 128x128")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

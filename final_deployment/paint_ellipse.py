#!/usr/bin/env python
# File: final_deployment/paint_ellipse.py
# Author: Denis Kurka
# Year: 2025
# License: CC0


import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# CONFIGURABLE CONSTANTS
# -----------------------------------------------------------------------------
FITTED_ELLIPSE_DIR = "./fitted_ellipse_outlines"   # Folder with the fitted (outline) ellipse masks
CROPPED_DIR        = "./cropped"                   # Folder with the cropped images
OUTPUT_FINAL_DIR   = "./final"                     # Where we'll save the final painted images

# Paint color
RED_COLOR = (255, 0, 0)

# -----------------------------------------------------------------------------
def paint_ellipse_in_red(cropped_img: Image.Image, ellipse_mask: Image.Image):
    """
    Paints the white pixels (255) from `ellipse_mask` onto `cropped_img` in red.
    Returns a new PIL Image in 'RGB'.

    1) Convert `cropped_img` to 'RGB' if it's not already.
    2) Resample `ellipse_mask` to match `cropped_img.size` (using NEAREST neighbor).
    3) Wherever ellipse_mask == 255, paint those pixels red in the output.
    """
    # Ensure cropped_img is RGB
    if cropped_img.mode != 'RGB':
        cropped_img = cropped_img.convert('RGB')

    # Convert to numpy arrays
    cropped_arr = np.array(cropped_img)  # shape=(H,W,3)
    # Resize ellipse_mask to the same size as cropped_img
    if ellipse_mask.size != cropped_img.size:
        ellipse_mask = ellipse_mask.resize(cropped_img.size, Image.NEAREST)

    ellipse_arr = np.array(ellipse_mask)  # shape=(H,W), 0 or 255

    # Paint red where ellipse_arr == 255
    # We'll make a boolean mask
    white_pixels = (ellipse_arr == 255)
    # Assign RED_COLOR
    cropped_arr[white_pixels] = RED_COLOR

    # Convert back to PIL Image
    final_img = Image.fromarray(cropped_arr, mode='RGB')
    return final_img

# -----------------------------------------------------------------------------
def paint_ellipse():
    ellipse_path = "./data/ellipse_fitted.png"
    cropped_path = "./data/data_cropped_final.png"

    # Load images
    ellipse_img = Image.open(ellipse_path).convert('L')  # black/white
    cropped_img = Image.open(cropped_path)               # grayscale or color

    # Paint the ellipse
    final_img = paint_ellipse_in_red(cropped_img, ellipse_img)

    # Save result
    out_path = "./data/final.png"
    final_img.save(out_path)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    paint_ellipse()

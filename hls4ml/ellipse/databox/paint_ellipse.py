#!/usr/bin/env python
# File: -----------------------------------------------------------------------------
# Author: Denis Kurka
# Year: 2025
# License: CC0


import os
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
def process_entire_folder(
    ellipse_dir=FITTED_ELLIPSE_DIR,
    cropped_dir=CROPPED_DIR,
    output_dir=OUTPUT_FINAL_DIR
):
    """
    For each .png file in `ellipse_dir`, find the matching image in `cropped_dir`,
    resize the ellipse to the cropped image's size, paint it in red, and save
    the result (same filename) in `output_dir`.
    """
    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect ellipse mask files
    ellipse_files = sorted([
        f for f in os.listdir(ellipse_dir)
        if f.lower().endswith('.png')
    ])
    if not ellipse_files:
        print(f"No PNG files found in '{ellipse_dir}'.")
        return

    for idx, fname in enumerate(ellipse_files, start=1):
        ellipse_path = os.path.join(ellipse_dir, fname)
        print(f"[{idx}/{len(ellipse_files)}] Painting ellipse on: {ellipse_path}")

        # Find matching cropped image by filename
        # (If your naming differs, adapt this logic)
        cropped_path = os.path.join(cropped_dir, fname)
        if not os.path.exists(cropped_path):
            print(f"  => No matching cropped image found for '{fname}'. Skipping.")
            continue

        # Load images
        ellipse_img = Image.open(ellipse_path).convert('L')  # black/white
        cropped_img = Image.open(cropped_path)               # grayscale or color

        # Paint the ellipse
        final_img = paint_ellipse_in_red(cropped_img, ellipse_img)

        # Save result
        out_path = os.path.join(output_dir, fname)
        final_img.save(out_path)
        print(f"  => Painted image saved to: {out_path}")

    print("\nAll done.")

# -----------------------------------------------------------------------------
def main():
    print("Painting fitted ellipses over cropped images in red...")
    process_entire_folder(FITTED_ELLIPSE_DIR, CROPPED_DIR, OUTPUT_FINAL_DIR)
    print("Done.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()


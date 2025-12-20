#!/usr/bin/env python
# File: -----------------------------------------------------------------------------
# Author: Denis Kurka
# Year: 2025
# License: CC0


import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import scipy.ndimage as ndi

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Folders
INPUT_ELLIPSES_DIR    = "./cropped_ellipses"    # Where the original cropped ellipse masks are
OUTPUT_SHRUNK_DIR     = "./shrunk_ellipses"     # Output folder for shrunk ellipses
OUTPUT_OUTLINE_DIR    = "./ellipse_outlines"    # Output folder for the ellipse circumference

# Morphological parameters
SHRINK_PIXELS  = 5   # Number of erosion iterations for shrinking
OUTLINE_WIDTH  = 3   # Pixel thickness for the circumference

# Debug
DEBUG_VISUALIZE = True
DEBUG_VIS_DIR   = "./debug_ellipses"  # Where to save debug images

# -----------------------------------------------------------------------------
def shrink_ellipse_mask(mask: np.ndarray, shrink_pixels: int) -> np.ndarray:
    """
    Perform a binary erosion on the ellipse (white=255, black=0) to 'shrink' it.
    shrink_pixels determines how many erosion iterations we apply.
    Returns a new mask as uint8 (0 or 255).
    """
    # Convert mask to boolean
    mask_bool = (mask > 0)
    # Erode
    eroded = ndi.binary_erosion(mask_bool, iterations=shrink_pixels)
    # Convert back to 0/255
    return (eroded.astype(np.uint8) * 255)

def outline_ellipse_mask(mask: np.ndarray, outline_width: int) -> np.ndarray:
    """
    Create an 'outline' (circumference) of the ellipse with the given pixel width.
    1) The 'boundary' = pixels in mask but not in its 1-iteration erosion.
    2) If outline_width>1, we dilate that boundary (outline_width - 1) times to make it thicker.
    Returns a uint8 mask with 0/255.
    """
    # Convert mask to boolean
    mask_bool = (mask > 0)

    # Single erosion for boundary
    eroded = ndi.binary_erosion(mask_bool, iterations=1)
    boundary = mask_bool & ~eroded  # difference

    # If we want a thicker boundary, we can dilate
    if outline_width > 1:
        boundary = ndi.binary_dilation(boundary, iterations=(outline_width - 1))

    return (boundary.astype(np.uint8) * 255)

def debug_visualization(
    original_mask: np.ndarray,
    shrunk_mask: np.ndarray,
    outline_mask: np.ndarray,
    out_path: str
):
    """
    Produce a debug figure with three subplots:
      1) Original ellipse mask
      2) Shrunk ellipse
      3) Outline ellipse
    Saves to out_path (PNG). No interactive pop-up.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(original_mask, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Original Ellipse")
    axs[0].axis("off")

    axs[1].imshow(shrunk_mask, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title(f"Shrunk by {SHRINK_PIXELS} px")
    axs[1].axis("off")

    axs[2].imshow(outline_mask, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title(f"Outline width={OUTLINE_WIDTH}")
    axs[2].axis("off")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def process_entire_folder(
    input_dir=INPUT_ELLIPSES_DIR,
    shrunk_dir=OUTPUT_SHRUNK_DIR,
    outline_dir=OUTPUT_OUTLINE_DIR
):
    """
    Reads every PNG in 'input_dir', creates:
      1) A shrunk ellipse (binary erosion).
      2) An outline ellipse (circumference) of a certain thickness.
    Saves them (with the same filename) into shrunk_dir and outline_dir, respectively.
    Optionally, produce a debug image side-by-side in DEBUG_VIS_DIR.
    """

    # Ensure output folders exist
    if not os.path.exists(shrunk_dir):
        os.makedirs(shrunk_dir)
    if not os.path.exists(outline_dir):
        os.makedirs(outline_dir)
    if DEBUG_VISUALIZE and not os.path.exists(DEBUG_VIS_DIR):
        os.makedirs(DEBUG_VIS_DIR)

    # Find all PNG masks
    ellipse_files = sorted([
        f for f in os.listdir(input_dir) if f.lower().endswith(".png")
    ])
    if not ellipse_files:
        print(f"No PNG files found in {input_dir}.")
        return

    for idx, fname in enumerate(ellipse_files, start=1):
        in_path = os.path.join(input_dir, fname)
        print(f"[{idx}/{len(ellipse_files)}] Processing: {in_path}")

        # Load mask (grayscale 0 or 255)
        mask = np.array(Image.open(in_path).convert("L"))  # shape=(H,W)

        # 1) Shrink
        shrunk = shrink_ellipse_mask(mask, SHRINK_PIXELS)

        # 2) Outline
        outlined = outline_ellipse_mask(mask, OUTLINE_WIDTH)

        # Save them with same filename, but in different output folders
        out_shrunk_path = os.path.join(shrunk_dir, fname)   # same name
        out_outline_path = os.path.join(outline_dir, fname) # same name

        Image.fromarray(shrunk).save(out_shrunk_path)
        Image.fromarray(outlined).save(out_outline_path)

        print(f"  => Shrunk ellipse -> {out_shrunk_path}")
        print(f"  => Outline ellipse -> {out_outline_path}")

        # Debug
        if DEBUG_VISUALIZE:
            debug_name = os.path.splitext(fname)[0] + "_debug.png"
            debug_path = os.path.join(DEBUG_VIS_DIR, debug_name)
            debug_visualization(mask, shrunk, outlined, debug_path)
            print(f"  => Debug figure saved to {debug_path}")

def main():
    print("Shrinking and outlining ellipses...")
    process_entire_folder(
        input_dir=INPUT_ELLIPSES_DIR,
        shrunk_dir=OUTPUT_SHRUNK_DIR,
        outline_dir=OUTPUT_OUTLINE_DIR
    )
    print("Done.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python
# File: hls4ml/ellipse/databox/crop_rect_from_original.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ORIG_H = 128         # The original image height
ORIG_W = 128         # The original image width
NET_H  = 128         # The network's resized height (where the rect masks come from)
NET_W  = 128         # The network's resized width

# Paths
ORIGINAL_IMAGES_DIR = "./original_images"       # Folder with the full-size original images
RECT_MASKS_DIR      = "./rectangle_masks"       # Folder with the rectangle-only masks (128x128)
CROPPED_OUT_DIR     = "./cropped"               # Where to save the cropped original images

# Paths for ellipses
EL_IMAGES_DIR = "./original_ellipse_masks"       # Folder with the full-size original images
CROPPED_EL_OUT_DIR     = "./cropped_ellipses"               # Where to save the cropped original images

# Paths for predictions
PRED_IMAGES_DIR = "./prediction_masks"       # Folder with the full-size original images
CROPPED_PRED_OUT_DIR     = "./cropped_prediction"               # Where to save the cropped original images

# Debug visualization
DEBUG_VISUALIZE = True
DEBUG_VIS_DIR   = "./debug_vis_crop"
DEBUG_VIS_EL_DIR   = "./debug_vis_el_crop"
DEBUG_VIS_PRED_DIR   = "./debug_vis_pred_crop"

# -----------------------------------------------------------------------------
def scale_mask_to_original(mask_2d: np.ndarray, orig_h: int, orig_w: int):
    """
    Scale the 2D mask (mask_2d) from shape (NET_H x NET_W) to (orig_h x orig_w).
    We use nearest-neighbor to keep the mask crisp (0 or 255).
    Returns a new 2D numpy array of shape (orig_h x orig_w).
    """
    # Use PIL for nearest-neighbor scaling
    # mask_2d is assumed to be np.uint8 (0 or 255)
    pil_mask_small = Image.fromarray(mask_2d, mode='L')
    pil_mask_large = pil_mask_small.resize((orig_w, orig_h), resample=Image.NEAREST)
    return np.array(pil_mask_large)

# -----------------------------------------------------------------------------
def get_mask_bounding_box(scaled_mask: np.ndarray):
    """
    Given a 2D mask (shape=(orig_h, orig_w)), find the bounding box of the nonzero (white) pixels.
    Returns (xmin, ymin, xmax, ymax). If there are no white pixels, returns None.
    """
    # We'll do a simple np.where
    rows, cols = np.where(scaled_mask > 0)
    if len(rows) == 0:
        # No white pixels
        return None
    ymin, ymax = rows.min(), rows.max()
    xmin, xmax = cols.min(), cols.max()
    return (xmin, ymin, xmax, ymax)

# -----------------------------------------------------------------------------
def crop_image_by_box(orig_image: Image.Image, box):
    """
    Crop the PIL 'orig_image' by the bounding box (xmin, ymin, xmax, ymax).
    Note that PIL's crop box is (left, upper, right, lower).
    We must add +1 if we want to include the boundary pixel.

    Returns the cropped PIL image.
    """
    (xmin, ymin, xmax, ymax) = box
    # PIL crop => (left, upper, right, lower), exclusive of right/lower
    return orig_image.crop((xmin, ymin, xmax+1, ymax+1))

# -----------------------------------------------------------------------------
def debug_visualize(orig_img: Image.Image, scaled_mask: np.ndarray, box, cropped_img: Image.Image, out_path: str):
    """
    Creates a debug figure to show:
      1) The original image (grayscale or color) with bounding box outline
      2) The upscaled rectangle mask
      3) The cropped sub-image

    Saves the figure to out_path; does not show interactively.
    """
    # Convert original image to displayable np array
    orig_arr = np.array(orig_img)

    # If grayscale, shape=(H, W); if color, shape=(H, W, 3)
    if len(orig_arr.shape) == 2:  # grayscale
        is_color = False
    else:
        is_color = True

    # Build an overlay on the original image
    overlay_arr = orig_arr.copy()

    if box is not None:
        (xmin, ymin, xmax, ymax) = box
        # Draw a red rectangle on overlay
        # We'll handle both grayscale and color
        if not is_color:
            # For grayscale, convert to 3-channel so we can color
            overlay_arr = np.stack([overlay_arr]*3, axis=-1)

        # Mark the bounding box in red
        overlay_arr[ymin:ymax+1, [xmin, xmax], :] = [255, 0, 0]
        overlay_arr[[ymin, ymax], xmin:xmax+1, :] = [255, 0, 0]

    # We'll create 1 row, 3 columns figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1) Original with bounding box
    axes[0].imshow(overlay_arr, cmap=None if is_color else 'gray', vmin=0, vmax=255)
    axes[0].set_title("Original w/ Box")
    axes[0].axis("off")

    # 2) Upscaled mask
    axes[1].imshow(scaled_mask, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Upscaled Rect Mask")
    axes[1].axis("off")

    # 3) Cropped sub-image
    cropped_arr = np.array(cropped_img)
    if len(cropped_arr.shape) == 2:  # grayscale
        axes[2].imshow(cropped_arr, cmap='gray', vmin=0, vmax=255)
    else:
        axes[2].imshow(cropped_arr)
    axes[2].set_title("Cropped Region")
    axes[2].axis("off")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# -----------------------------------------------------------------------------
def process_entire_folder(
    original_dir=ORIGINAL_IMAGES_DIR,
    rect_masks_dir=RECT_MASKS_DIR,
    output_dir=CROPPED_OUT_DIR,
    orig_h=ORIG_H,
    orig_w=ORIG_W,
    net_h=NET_H,
    net_w=NET_W,
    debug_dir=DEBUG_VIS_DIR
):
    """
    For each file in `rect_masks_dir` (PNG masks), we:
      1. Identify the matching original image in `original_dir`.
         (We assume the filenames match, e.g. 'image1.png' in both.)
      2. Upscale the mask from (net_h x net_w) -> (orig_h x orig_w) using nearest neighbor.
      3. Find the bounding box of the white area in the scaled mask.
      4. Crop the original image accordingly.
      5. Save the cropped region to `output_dir`.
      6. (Optional) If DEBUG_VISUALIZE, also produce a debug figure with side-by-side comparisons.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if DEBUG_VISUALIZE and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # Gather rectangle mask files
    mask_files = sorted([
        f for f in os.listdir(rect_masks_dir)
        if f.lower().endswith(".png")
    ])

    if not mask_files:
        print(f"No .png files found in {rect_masks_dir}.")
        return

    for idx, mask_fname in enumerate(mask_files, start=1):
        mask_path = os.path.join(rect_masks_dir, mask_fname)
        print(f"\n[{idx}/{len(mask_files)}] Processing: {mask_path}")

        # Load rectangle mask (which is net_h x net_w)
        rect_mask = np.array(Image.open(mask_path).convert('L'))  # shape=(net_h, net_w), 0 or 255

        # Upscale to original size
        scaled_mask = scale_mask_to_original(rect_mask, orig_h, orig_w)  # shape=(orig_h, orig_w)

        # Figure out matching original image path. Assuming same filename.
        # e.g. if mask_fname = "image1_rect.png" => original might be "image1.png"
        # Adjust this logic to your real naming scheme:
        base_name, _ = os.path.splitext(mask_fname)

        # Find the first alphabetical character to split the name before it
        first_alpha_index = next((i for i, c in enumerate(base_name) if c == '_'), len(base_name))
        base_name = base_name[:first_alpha_index]

        # Generate candidates from the original_dir
        orig_candidates = []
        for f in os.listdir(original_dir):
            orig_path, _ = os.path.splitext(f)
            first_alpha_index = next((i for i, c in enumerate(orig_path) if c == '_'), len(orig_path))
            orig_path_trimmed = orig_path[:first_alpha_index]
            if orig_path_trimmed == base_name:
                orig_candidates.append(f)

        if not orig_candidates:
            print(f"  => No matching original image found for {mask_fname}, base name: {base_name}, skipping.")
            continue
        # If multiple matches, just pick the first:
        orig_image_fname = orig_candidates[0]
        orig_image_path = os.path.join(original_dir, orig_image_fname)

        print(f"  => Matching original: {orig_image_path}")
        orig_img = Image.open(orig_image_path)

        # Ensure the original image is indeed (orig_h x orig_w)
        if orig_img.size != (orig_w, orig_h):
            print(f"  => Warning: original image size {orig_img.size} != ({orig_w},{orig_h}). Using as-is.")

        # Get bounding box of white region in scaled_mask
        bbox = get_mask_bounding_box(scaled_mask)
        if not bbox:
            print("  => No white pixels found in mask. Skipping.")
            continue

        # Crop the original image
        cropped_img = crop_image_by_box(orig_img, bbox)

        # Save
        out_fname = base_name + "_cropped.png"
        out_path = os.path.join(output_dir, out_fname)
        cropped_img.save(out_path)
        print(f"  => Cropped region saved to {out_path}")

        # Debug visualize?
        if DEBUG_VISUALIZE:
            debug_fname = base_name + "_debug_crop.png"
            debug_path = os.path.join(debug_dir, debug_fname)
            debug_visualize(orig_img, scaled_mask, bbox, cropped_img, debug_path)
            print(f"  => Debug figure saved to {debug_path}")

# -----------------------------------------------------------------------------
def main():
    print("Running 'crop_by_rect_mask' script...")
#    process_entire_folder(
#        original_dir=ORIGINAL_IMAGES_DIR,
#        rect_masks_dir=RECT_MASKS_DIR,
#        output_dir=CROPPED_OUT_DIR,
#        orig_h=ORIG_H,
#        orig_w=ORIG_W,
#        net_h=NET_H,
#        net_w=NET_W,
#        debug_dir=DEBUG_VIS_DIR
#    )
#
#    process_entire_folder(
#        original_dir=EL_IMAGES_DIR,
#        rect_masks_dir=RECT_MASKS_DIR,
#        output_dir=CROPPED_EL_OUT_DIR,
#        orig_h=ORIG_H,
#        orig_w=ORIG_W,
#        net_h=NET_H,
#        net_w=NET_W,
#        debug_dir=DEBUG_VIS_EL_DIR
#    )

    # THIS REQUIRES DIFFERENT RESOLUTION!!! ORIGINAL IS 128x128
    process_entire_folder(
        original_dir=PRED_IMAGES_DIR,
        rect_masks_dir=RECT_MASKS_DIR,
        output_dir=CROPPED_PRED_OUT_DIR,
        orig_h=ORIG_H,
        orig_w=ORIG_W,
        net_h=NET_H,
        net_w=NET_W,
        debug_dir=DEBUG_VIS_PRED_DIR
    )
    print("Done.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

#!/usr/bin/env python
# File: hls4ml/ellipse/databox/rects_from_masks.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
RECT_W = 32
RECT_H = 32

INPUT_MASKS_DIR = "./prediction_masks"      # Input folder with predicted masks (PNG)
OUTPUT_MASKS_DIR = "./rectangle_masks"      # Output folder for final rectangle-only masks
DEBUG_VISUALIZE = True                      # If True, produce debug visualization images
DEBUG_VIS_DIR = "./debug_vis"               # Where to save debug images
ALPHA = 0.4                                 # Transparency for overlay rectangle

# -------------------------------------------------------------------------
#def find_best_rectangle(mask_array: np.ndarray, rect_w: int, rect_h: int):
#    """
#    Slide a (rect_w x rect_h) window across mask_array (2D, shape=(H, W)) and find the
#    sub-region that yields the greatest sum of pixel values. Returns (best_x,best_y) plus that sum.
#    """
#    height, width = mask_array.shape
#    best_sum = -1
#    best_coords = (0, 0)
#
#    for y in range(height - rect_h + 1):
#        for x in range(width - rect_w + 1):
#            region_sum = np.sum(mask_array[y:y + rect_h, x:x + rect_w])
#            if region_sum > best_sum:
#                best_sum = region_sum
#                best_coords = (x, y)
#
#    return best_coords, best_sum

def find_best_rectangle(mask_array: np.ndarray, rect_w: int, rect_h: int):
    """
    A more advanced approach that:
      1. Finds the largest "white" connected region in `mask_array` (2D),
         where "white" means any pixel > 0.
      2. Computes the intensity-weighted centroid of that region.
      3. Places a rect_w x rect_h rectangle so that its center is at that centroid.
      4. Returns ((best_x, best_y), best_sum), where best_sum is the sum of
         pixel intensities inside that rectangle.

    NOTE: This is not a naive sliding window. Instead, we:
      - Label connected components via BFS (8-connectivity).
      - Pick the component with the highest total intensity sum.
      - Compute that component's centroid (x_center, y_center) = sum(x*val)/sum(val), sum(y*val)/sum(val).
      - Align the rectangle center to that centroid (clamped within image bounds).
      - Calculate the sum of pixel values in that rectangle.

    Same interface: returns ( (best_x, best_y), best_sum ).
    """
    import collections

    H, W = mask_array.shape
    visited = np.zeros((H, W), dtype=bool)

    # Track the best connected component by "sum of pixel intensities."
    best_label_sum  = 0
    best_label_sumx = 0  # sum of (x * pixel_value)
    best_label_sumy = 0  # sum of (y * pixel_value)

    # 8-direction neighbors for BFS
    neighbors8 = [
        (-1, -1), (-1,  0), (-1,  1),
        ( 0, -1),           ( 0,  1),
        ( 1, -1), ( 1,  0), ( 1,  1)
    ]

    # ----------------------------------------------------------------------
    # 1. Find the largest-intensity connected region (BFS)
    # ----------------------------------------------------------------------
    for row in range(H):
        for col in range(W):
            val = mask_array[row, col]
            # If pixel is > 0 and not yet visited, BFS it
            if val > 0 and not visited[row, col]:
                queue = collections.deque()
                queue.append((row, col))
                visited[row, col] = True

                # We'll accumulate the sum of intensities, plus sum of x*val, y*val for centroid
                comp_sum  = 0
                comp_sumx = 0
                comp_sumy = 0

                while queue:
                    r, c = queue.popleft()
                    pv = mask_array[r, c]  # pixel value
                    comp_sum  += pv
                    comp_sumx += (c * pv)
                    comp_sumy += (r * pv)

                    # Check neighbors
                    for dr, dc in neighbors8:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if (mask_array[nr, nc] > 0) and (not visited[nr, nc]):
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                # After BFS, check if this component is our "largest sum"
                if comp_sum > best_label_sum:
                    best_label_sum  = comp_sum
                    best_label_sumx = comp_sumx
                    best_label_sumy = comp_sumy

    # ----------------------------------------------------------------------
    # 2. If no white pixels at all, default to (0,0) with sum=0
    # ----------------------------------------------------------------------
    if best_label_sum <= 0:
        return ( (0, 0), 0 )

    # ----------------------------------------------------------------------
    # 3. Compute the intensity-weighted centroid of that largest region
    # ----------------------------------------------------------------------
    cx = best_label_sumx / best_label_sum  # Weighted average x
    cy = best_label_sumy / best_label_sum  # Weighted average y

    # ----------------------------------------------------------------------
    # 4. Center the rectangle at (cx, cy), clamp within the image
    # ----------------------------------------------------------------------
    half_w = rect_w // 2
    half_h = rect_h // 2

    best_x = int(round(cx - half_w))
    best_y = int(round(cy - half_h))

    # Clamp to the image boundaries
    if best_x < 0:
        best_x = 0
    if best_y < 0:
        best_y = 0
    if best_x + rect_w > W:
        best_x = W - rect_w
    if best_y + rect_h > H:
        best_y = H - rect_h

    # ----------------------------------------------------------------------
    # 5. Compute the sum of pixel intensities inside that rectangle
    # ----------------------------------------------------------------------
    best_sum_region = np.sum(mask_array[best_y:best_y + rect_h, best_x:best_x + rect_w])

    # Return the same info as always: coordinates + sum in that rectangle
    return ( (best_x, best_y), best_sum_region )


# -------------------------------------------------------------------------
def create_binary_rectangle_mask(
    height: int,
    width: int,
    best_x: int,
    best_y: int,
    rect_w: int,
    rect_h: int
):
    """
    Create a new 2D mask (height x width) with 0 (black) everywhere,
    except for a white (255) rectangle at (best_x, best_y) of size (rect_w x rect_h).
    """
    new_mask = np.zeros((height, width), dtype=np.uint8)
    new_mask[best_y:best_y + rect_h, best_x:best_x + rect_w] = 255
    return new_mask

# -------------------------------------------------------------------------
def debug_visualization(original_mask: np.ndarray, rect_mask: np.ndarray, out_path: str, alpha=0.4):
    """
    Creates a single image (saved as PNG) that shows:
      1) The original predicted mask (grayscale)
      2) The rectangle-only mask
      3) An overlay: the original mask with the rectangle highlighted in red with alpha transparency.
    No interactive window is shown; we only save the figure to `out_path`.
    """
    # original_mask, rect_mask => shape (H, W), 0..255
    # Make sure they are the same shape
    H, W = original_mask.shape
    if rect_mask.shape != (H, W):
        raise ValueError("rect_mask shape does not match original_mask shape.")

    # Prepare 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # 1) Original predicted mask
    axs[0].imshow(original_mask, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Original Predicted Mask")
    axs[0].axis("off")

    # 2) Rectangle-only mask
    axs[1].imshow(rect_mask, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Rectangle-Only Mask")
    axs[1].axis("off")

    # 3) Overlay the rectangle on the original (in red)
    #    We'll build an RGB version of the original
    overlay_rgb = np.dstack([original_mask, original_mask, original_mask]).astype(np.float32)
    # Where rect_mask == 255, we blend with red color [255, 0, 0]
    red = np.array([255, 0, 0], dtype=np.float32)
    mask_indices = (rect_mask == 255)

    overlay_rgb[mask_indices] = (
        overlay_rgb[mask_indices] * (1 - alpha) + red * alpha
    )
    overlay_rgb = overlay_rgb.astype(np.uint8)  # convert back to 8-bit

    axs[2].imshow(overlay_rgb, vmin=0, vmax=255)
    axs[2].set_title("Overlay (red box)")
    axs[2].axis("off")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)  # Close to avoid GUI

# -------------------------------------------------------------------------
def process_entire_folder(
    input_dir=INPUT_MASKS_DIR,
    output_dir=OUTPUT_MASKS_DIR,
    rect_w=RECT_W,
    rect_h=RECT_H
):
    """
    Reads all PNG masks in input_dir, finds the best rect_w x rect_h region in each,
    and saves a new mask in output_dir that has a single white rectangle (255)
    at that location (everything else is black). Optionally creates debug images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if DEBUG_VISUALIZE and not os.path.exists(DEBUG_VIS_DIR):
        os.makedirs(DEBUG_VIS_DIR)

    # Find all PNG files in input_dir
    mask_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".png")]
    if not mask_files:
        print(f"No .png files found in '{input_dir}'.")
        return

    for idx, fname in enumerate(mask_files, start=1):
        in_path = os.path.join(input_dir, fname)
        print(f"[{idx}/{len(mask_files)}] Processing: {in_path}")

        # Load the mask as grayscale (0..255)
        with Image.open(in_path) as img:
            mask_gray = img.convert("L")
        mask_array = np.array(mask_gray)  # shape: (H, W)

        # Find the best rectangle
        (best_x, best_y), best_sum = find_best_rectangle(mask_array, rect_w, rect_h)
        # Create a new 2D mask that is black except for a 255 rectangle
        H, W = mask_array.shape
        rect_mask = create_binary_rectangle_mask(H, W, best_x, best_y, rect_w, rect_h)

        # Save the rectangle-only mask
        out_name = os.path.splitext(fname)[0] + "_rect.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(rect_mask).save(out_path)
        print(f"   => Best sum={best_sum}, coords=({best_x},{best_y}), saved to {out_path}")

        # Debug Visualization (if enabled)
        if DEBUG_VISUALIZE:
            debug_img_name = os.path.splitext(fname)[0] + "_debug.png"
            debug_img_path = os.path.join(DEBUG_VIS_DIR, debug_img_name)
            debug_visualization(mask_array, rect_mask, debug_img_path, alpha=ALPHA)
            print(f"   [Debug] Visualization saved to {debug_img_path}")

# -------------------------------------------------------------------------
def main():
    process_entire_folder(
        input_dir=INPUT_MASKS_DIR,
        output_dir=OUTPUT_MASKS_DIR,
        rect_w=RECT_W,
        rect_h=RECT_H
    )

if __name__ == "__main__":
    main()

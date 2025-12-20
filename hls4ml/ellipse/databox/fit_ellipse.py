#!/usr/bin/env python

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# scikit-image for regionprops and drawing
from skimage import measure, draw

# -----------------------------------------------------------------------------
# CONFIGURABLE CONSTANTS
# -----------------------------------------------------------------------------
INPUT_MASKS_DIR    = "./prediction_ellipse"
OUTPUT_ELLIPSE_DIR = "./fitted_ellipse_outlines"

DEBUG_VISUALIZE = True
DEBUG_VIS_DIR   = "./debug_fitted_ellipses_outline"

# -----------------------------------------------------------------------------
def fit_ellipse_outline_around_mask(mask: np.ndarray) -> np.ndarray:
    """
    Given a 2D binary mask (values 0 or 255) containing some 'ellipse-like' shape,
    fit a best ellipse using image moments (regionprops), then draw **only the outline**
    (perimeter) of that fitted ellipse. The outline is a single pixel wide.

    Returns a new 2D array, same shape as input, with the fitted ellipse perimeter
    drawn in white (255). Everything else is black (0).

    Steps:
      1) Convert to boolean: (mask > 0).
      2) Label the image & pick the largest labeled region.
      3) Use regionprops to get:
           - centroid (y, x)
           - orientation (radians; angle from vertical axis)
           - major_axis_length
           - minor_axis_length
      4) Convert the regionprops orientation to the format expected by skimage.draw.ellipse_perimeter.
      5) Draw the ellipse perimeter in the output mask.
    """
    bw = (mask > 0)

    # Label and pick the largest region
    labeled = measure.label(bw)
    regions = measure.regionprops(labeled)
    if not regions:
        # No foreground -> return empty
        return np.zeros_like(mask, dtype=np.uint8)

    # Sort by area, pick largest
    regions.sort(key=lambda r: r.area, reverse=True)
    region = regions[0]

    y0, x0 = region.centroid
    orientation = region.orientation  # regionprops: angle from vertical (rows) in [-pi/2, pi/2]
    major_len = region.major_axis_length
    minor_len = region.minor_axis_length

    # If no significant region
    if major_len < 1 or minor_len < 1:
        return np.zeros_like(mask, dtype=np.uint8)

    # Build an empty output
    out_mask = np.zeros_like(mask, dtype=np.uint8)

    # ellipse_perimeter expects:
    #   r (row center), c (col center),
    #   r_radius, c_radius,
    #   orientation in [0, 2*pi), measured CCW from horizontal.
    #
    # regionprops orientation=0 => ellipse is vertical => we want ellipse_perimeter orientation= pi/2
    # We'll set the ellipse radius along rows to be half the major axis if orientation=0 => vertical,
    # which is consistent with regionprops. So:
    r_radius = major_len / 2.0
    c_radius = minor_len / 2.0

    # Transform regionprops orientation => ellipse_perimeter orientation
    # Because regionprops orientation=0 means major axis is vertical,
    # whereas ellipse_perimeter orientation=0 means major axis is horizontal.
    ellipse_orientation = np.pi/2 - orientation

    # We must provide integer centers and integer radii to ellipse_perimeter (round them).
    rr, cc = draw.ellipse_perimeter(
        int(round(y0)), 
        int(round(x0)),
        int(round(r_radius)), 
        int(round(c_radius)),
        orientation=ellipse_orientation,
        shape=out_mask.shape
    )

    # Draw the outline as white (255)
    out_mask[rr, cc] = 255

    return out_mask

# -----------------------------------------------------------------------------
def debug_visualization(original_mask: np.ndarray, fitted_outline: np.ndarray, out_path: str):
    """
    Debug figure with side-by-side subplots:
      1) Original predicted mask
      2) Fitted ellipse OUTLINE only
    Saves to out_path.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(original_mask, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Original Predicted Mask")
    axs[0].axis("off")

    axs[1].imshow(fitted_outline, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Fitted Ellipse Outline")
    axs[1].axis("off")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# -----------------------------------------------------------------------------
def process_entire_folder(
    input_dir=INPUT_MASKS_DIR,
    output_dir=OUTPUT_ELLIPSE_DIR
):
    """
    For each mask in `input_dir`:
      1. Load as 8-bit grayscale (0..255).
      2. Fit an ellipse perimeter around the largest region.
      3. Create a new mask with just that ellipse outline (single pixel wide).
      4. Save the result to `output_dir` with the same filename.
      5. If DEBUG_VISUALIZE, save a side-by-side debug figure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if DEBUG_VISUALIZE and not os.path.exists(DEBUG_VIS_DIR):
        os.makedirs(DEBUG_VIS_DIR)

    mask_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    if not mask_files:
        print(f"No PNG files found in '{input_dir}'.")
        return

    for idx, fname in enumerate(mask_files, start=1):
        in_path = os.path.join(input_dir, fname)
        print(f"[{idx}/{len(mask_files)}] Fitting ellipse outline to: {in_path}")

        orig_mask = np.array(Image.open(in_path).convert("L"))

        # Fit the perimeter
        outline_mask = fit_ellipse_outline_around_mask(orig_mask)

        # Save
        out_path = os.path.join(output_dir, fname)
        Image.fromarray(outline_mask).save(out_path)
        print(f"  => Saved fitted ellipse outline to {out_path}")

        # Debug
        if DEBUG_VISUALIZE:
            debug_name = os.path.splitext(fname)[0] + "_debug.png"
            debug_path = os.path.join(DEBUG_VIS_DIR, debug_name)
            debug_visualization(orig_mask, outline_mask, debug_path)
            print(f"  => Debug figure saved to {debug_path}")

    print("\nAll done.")

# -----------------------------------------------------------------------------
def main():
    print("Fitting ellipse outlines around predicted ellipse masks...")
    process_entire_folder(INPUT_MASKS_DIR, OUTPUT_ELLIPSE_DIR)
    print("Done.")

if __name__ == "__main__":
    main()

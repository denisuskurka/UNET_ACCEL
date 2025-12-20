#!/usr/bin/env python3
# File: flask_dataprep.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
flask_dataprep.py

Pairs images labeled with '-drawn' (in one folder) to their corresponding original images 
in another folder, extracts the boundary color into a filled mask, optionally shrinks it,
and saves the results.

USAGE:
  python flask_dataprep.py -a <DRAWN_FOLDER> -b <ORIGINAL_FOLDER> -c <MASKS_OUTPUT> -d <ORIGS_OUTPUT>
  python flask_dataprep.py --color "0,255,0" --shrink 10 --max_ratio 0.15 --comparisons_folder ./comparisons

EXAMPLE:
  python flask_dataprep.py \
    -a ./patricie \
    -b ./no_label \
    -c ./masks \
    -d ./originals \
    --color "0,255,0" \
    --shrink 10 \
    --max_ratio 0.15 \
    --comparisons_folder ./comparisons
    
PARAMETERS:
  -a, --drawn_folder:       Folder with labeled images (filenames contain '-drawn').
  -b, --original_folder:    Folder with unlabeled/original images.
  -c, --masks_folder:       Output folder for generated masks.
  -d, --originals_folder:   Output folder for cropped originals.
  --comparisons_folder:     Output folder for saving the 4-panel comparison figure.
  --tolerance:              Color tolerance for extraction (default=5).
  --color:                  BGR color for boundary extraction as "B,G,R" (default="0,128,0").
  --show_result:            Display the 4-panel figure on screen.
  --no_fill:                Skip the morphological fill step (for boundary closure).
  --shrink:                 Erode (shrink) the final mask by N pixels.
  --max_ratio:              If mask's white ratio > this threshold, skip saving (default=0.15 => 15%).
  --help-only:              Show this help message and exit.

When no arguments are provided, defaults will be used.
"""

import os
import cv2
import numpy as np
import argparse
import sys
import math
import matplotlib.pyplot as plt

# Adjust as needed
CROP_MARGIN = (math.floor(80*0.64), math.floor(20*0.64), 0, math.floor(185*0.64))

def crop_image(image, margin=CROP_MARGIN):
    """
    Crops the image by 'margin' on each side. 'margin' can be:
      1) An integer, which means crop that many pixels from every side.
      2) A 4-tuple of the form (top, bottom, left, right).

    Returns:
      The cropped image. If the margins are too large (width or height <= 0),
      it returns the original image.
    """
    h, w = image.shape[:2]

    if isinstance(margin, int):
        top = bottom = left = right = margin
    elif isinstance(margin, (tuple, list)) and len(margin) == 4:
        top, bottom, left, right = margin
    else:
        raise ValueError("margin must be either an integer or a 4-tuple (top, bottom, left, right).")

    new_top = top
    new_bottom = h - bottom
    new_left = left
    new_right = w - right

    if new_top >= new_bottom or new_left >= new_right:
        print(f"Warning: cannot crop with margin {margin} from a {w}x{h} image. Returning original.")
        return image
    
    return image[new_top:new_bottom, new_left:new_right]


def fill_mask(mask, close_gaps_kernel=10):
    """
    1) Apply a morphological CLOSE to fix small breaks in the boundary.
    2) Fill the interior so the entire object is white (255).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_gaps_kernel, close_gaps_kernel))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(closed_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled_mask


def shrink_mask(mask, shrink_pixels=0):
    """
    Erodes (shrinks) the white region of 'mask' by 'shrink_pixels'.
    A morphological erosion operation reduces the mask boundary inward.
    """
    if shrink_pixels <= 0:
        return mask  # No shrink

    ksize = 2 * shrink_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    return shrunk_mask


def compute_mask_ratio(mask):
    """
    Computes the ratio of white pixels to total pixels in 'mask'.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = cv2.countNonZero(mask)  # same as np.count_nonzero
    ratio = white_pixels / float(total_pixels)
    return ratio


def extract_specific_color(
    image, color_bgr, tolerance=20, fill_boundaries=False,
    shrink_pixels=0
):
    """
    Returns a single-channel binary mask (0 or 255) for the given color in 'image'.
    Steps:
      1) Threshold by color + tolerance => boundary mask
      2) Optionally fill boundary-only masks (with morphological close)
      3) Optionally shrink/erode the mask by 'shrink_pixels'
    """
    target_b, target_g, target_r = color_bgr
    lower_bound = np.array([
        max(target_b - tolerance, 0),
        max(target_g - tolerance, 0),
        max(target_r - tolerance, 0)
    ], dtype=np.uint8)
    upper_bound = np.array([
        min(target_b + tolerance, 255),
        min(target_g + tolerance, 255),
        min(target_r + tolerance, 255)
    ], dtype=np.uint8)
    
    mask = cv2.inRange(image, lower_bound, upper_bound)

    if fill_boundaries:
        mask = fill_mask(mask)

    if shrink_pixels > 0:
        mask = shrink_mask(mask, shrink_pixels=shrink_pixels)

    return mask


def save_comparison_figure(cropped_drawn, cropped_orig, mask, output_path):
    """
    Saves the 4-panel comparison (drawn, orig, mask, overlay) into the file `output_path`.
    """
    # Convert BGR -> RGB for matplotlib display
    drawn_rgb = cv2.cvtColor(cropped_drawn, cv2.COLOR_BGR2RGB)
    orig_rgb  = cv2.cvtColor(cropped_orig, cv2.COLOR_BGR2RGB)
    
    # Convert single-channel mask to BGR so we can overlay in color
    mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Blend with some alpha
    overlay_bgr = cv2.addWeighted(cropped_orig, 0.7, mask_bgr, 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(drawn_rgb)
    axes[0].set_title("Labeled (drawn)")
    axes[0].axis('off')

    axes[1].imshow(orig_rgb)
    axes[1].set_title("Unlabeled (orig)")
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Extracted Mask")
    axes[2].axis('off')

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Overlay (mask + orig)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)  # close to avoid GUI popups or memory usage


def show_comparison(cropped_drawn, cropped_orig, mask):
    """
    Displays the 4-panel figure on the screen (for debugging).
    """
    # Convert BGR -> RGB for matplotlib
    drawn_rgb = cv2.cvtColor(cropped_drawn, cv2.COLOR_BGR2RGB)
    orig_rgb  = cv2.cvtColor(cropped_orig, cv2.COLOR_BGR2RGB)
    
    # Convert single-channel mask to BGR for overlay
    mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay_bgr = cv2.addWeighted(cropped_orig, 0.7, mask_bgr, 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(drawn_rgb)
    axes[0].set_title("Labeled (drawn)")
    axes[0].axis('off')

    axes[1].imshow(orig_rgb)
    axes[1].set_title("Unlabeled (orig)")
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Extracted Mask")
    axes[2].axis('off')

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Overlay (mask + orig)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def process_drawn_images_with_separate_originals(
    drawn_folder,
    original_folder,
    color_bgr,
    tolerance=20,
    masks_folder=None,
    originals_folder=None,
    comparisons_folder=None,
    fill_boundaries=False,
    show_result=False,
    shrink_pixels=0,
    max_ratio=0.15
):
    """
    For each labeled image in 'drawn_folder' that has '-drawn' in its name:
      1) Remove '-drawn' to find the corresponding unlabeled file in 'original_folder'.
      2) If the unlabeled file doesn't exist, skip.
      3) Read both images (OpenCV). Resize 'orig_img' to match 'drawn_img' size.
      4) Crop both images (using CROP_MARGIN).
      5) Extract boundary mask from 'drawn_img':
         - threshold by color
         - optional fill
         - optional shrink
      6) Compute mask ratio (white/total). If ratio > max_ratio, skip saving.
      7) (Optional) Display results if show_result=True.
      8) (Optional) Save comparison figure if comparisons_folder is provided.
      9) Save mask to 'masks_folder' and cropped original to 'originals_folder'.
    """
    # Ensure output folders exist
    if masks_folder and not os.path.exists(masks_folder):
        os.makedirs(masks_folder, exist_ok=True)
    if originals_folder and not os.path.exists(originals_folder):
        os.makedirs(originals_folder, exist_ok=True)
    if comparisons_folder and not os.path.exists(comparisons_folder):
        os.makedirs(comparisons_folder, exist_ok=True)

    drawn_files = os.listdir(drawn_folder)
    
    for file_name in drawn_files:
        # We only care about files that contain '-drawn'
        if '-labeled' not in file_name.lower():
            continue
        
        drawn_path = os.path.join(drawn_folder, file_name)
        root, ext = os.path.splitext(file_name)
        unlabeled_root = root.replace('-labeled', '')
        original_filename = f"{unlabeled_root}{ext}"
        original_path = os.path.join(original_folder, original_filename)

        # If the original does not exist, skip
        if not os.path.isfile(original_path):
            print(f"No matching original for {file_name} at: {original_path} => skipping.")
            continue

        # Read images
        drawn_img = cv2.imread(drawn_path)
        orig_img = cv2.imread(original_path)

        if drawn_img is None:
            print(f"Failed to read labeled image: {drawn_path}, skipping.")
            continue
        if orig_img is None:
            print(f"Failed to read original image: {original_path}, skipping.")
            continue

        # 1) Resize 'orig_img' to match 'drawn_img' shape
        h_drawn, w_drawn = drawn_img.shape[:2]
        orig_img = cv2.resize(orig_img, (w_drawn, h_drawn), interpolation=cv2.INTER_AREA)

        # 2) Crop both images
        cropped_drawn = crop_image(drawn_img, CROP_MARGIN)
        cropped_orig = crop_image(orig_img, CROP_MARGIN)

        # 3) Extract boundary mask
        mask = extract_specific_color(
            cropped_drawn,
            color_bgr=color_bgr,
            tolerance=tolerance,
            fill_boundaries=fill_boundaries,
            shrink_pixels=shrink_pixels
        )

        # 4) Check mask ratio
        ratio = compute_mask_ratio(mask)  # e.g. 0.12 => 12%
        if ratio > max_ratio:
            print(f"Mask ratio {ratio*100:.2f}% exceeds max {max_ratio*100:.1f}%. Skipping {file_name}.")
            continue  # skip saving or showing

        # 5) Show the results if requested
        if show_result:
            show_comparison(cropped_drawn, cropped_orig, mask)

        # 6) Optionally save the comparison figure
        if comparisons_folder:
            comp_output_name = f"{unlabeled_root}_comparison.png"
            comp_output_path = os.path.join(comparisons_folder, comp_output_name)
            save_comparison_figure(cropped_drawn, cropped_orig, mask, comp_output_path)
            print(f"Saved comparison figure -> {comp_output_path}")

        # 7) Build output filenames for mask & original
        output_mask_name = f"{unlabeled_root}_mask.png"
        output_orig_name = f"{unlabeled_root}_orig.png"

        # 8) Save the mask
        if masks_folder:
            mask_output_path = os.path.join(masks_folder, output_mask_name)
            cv2.imwrite(mask_output_path, mask)
            print(f"Saved mask -> {mask_output_path}")

        # 9) Save the cropped original
        if originals_folder:
            orig_output_path = os.path.join(originals_folder, output_orig_name)
            cv2.imwrite(orig_output_path, cropped_orig)
            print(f"Saved original -> {orig_output_path}")

        print(f"Done pairing: {file_name} with {original_filename} [Mask ratio: {ratio*100:.2f}%]")


def main():
    parser = argparse.ArgumentParser(
        description="Extract color boundaries from '-drawn' images, pair with unlabeled originals, and save results.",
        add_help=False
    )

    parser.add_argument('-h', '--help', action='help',
                        help="Show this help message and exit.")

    parser.add_argument(
        "--help-only",
        action="store_true",
        help="Only print help message and exit."
    )
    parser.add_argument(
        "-a", "--drawn_folder",
        default="/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/finn/notebooks/FZ5-UNET/SN/new_raw_data/anezka/",
        help="Path to folder containing labeled images with '-drawn' in the filename."
    )
    parser.add_argument(
        "-b", "--original_folder",
        default="/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/finn/notebooks/FZ5-UNET/SN/new_raw_data_no_label/",
        help="Path to folder containing unlabeled/original images."
    )
    parser.add_argument(
        "-c", "--masks_folder",
        default="/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/finn/notebooks/FZ5-UNET/SN/new_raw_data_masks/",
        help="Path to output folder for generated masks."
    )
    parser.add_argument(
        "-d", "--originals_folder",
        default="/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/finn/notebooks/FZ5-UNET/SN/new_raw_data_originals/",
        help="Path to output folder for cropped originals."
    )
    parser.add_argument(
        "--comparisons_folder",
        default="./comparisons",
        help="Optional folder to store the 4-panel comparison figure (e.g. ./comparisons)."
    )
    parser.add_argument(
        "--tolerance", type=int, default=5,
        help="Color tolerance for extraction (default=5)."
    )
    parser.add_argument(
        "--color",
        default="0,164,255",
        help="BGR color for boundary extraction as 'B,G,R' (default='0,164,255')."
    )
    parser.add_argument(
        "--show_result", action="store_true",
        help="If set, display each result with 4 panels: [drawn, unlabeled, mask, overlay]."
    )
    parser.add_argument(
        "--no_fill", action="store_true",
        help="If set, skip the morphological fill step."
    )
    parser.add_argument(
        "--shrink", type=int, default=0,
        help="Erode (shrink) the mask by this many pixels (default=0)."
    )
    parser.add_argument(
        "--max_ratio", type=float, default=0.02,
        help="If mask's white ratio is higher than this fraction, skip saving. (default=0.15 => 15%%)"
    )

    args = parser.parse_args()

    if args.help_only:
        parser.print_help()
        sys.exit(0)

    # Parse the BGR color string, e.g. "0,128,0"
    try:
        b_str, g_str, r_str = args.color.split(",")
        color_bgr = (int(b_str.strip()), int(g_str.strip()), int(r_str.strip()))
    except ValueError:
        print(f"Error: --color argument must be in 'B,G,R' format. Got '{args.color}'.")
        sys.exit(1)

    fill_boundaries = not args.no_fill

    process_drawn_images_with_separate_originals(
        drawn_folder=args.drawn_folder,
        original_folder=args.original_folder,
        color_bgr=color_bgr,
        tolerance=args.tolerance,
        masks_folder=args.masks_folder,
        originals_folder=args.originals_folder,
        comparisons_folder=args.comparisons_folder,
        fill_boundaries=fill_boundaries,
        show_result=args.show_result,
        shrink_pixels=args.shrink,
        max_ratio=args.max_ratio
    )

    print("All operations have completed.")


if __name__ == "__main__":
    main()


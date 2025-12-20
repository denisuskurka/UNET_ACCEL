#!/usr/bin/env python
# File: scikit-image for regionprops and drawing
# Author: Denis Kurka
# Year: 2025
# License: CC0


import cv2
import numpy as np
from PIL import Image

# scikit-image for regionprops and drawing
from skimage import measure, draw

# -----------------------------------------------------------------------------
SHRINK_FACTOR = 1  # Adjust this between (0..1). Smaller => ellipse is more "inside".


def fit_ellipse_contour(mask: np.ndarray, shrink_factor=1.0) -> np.ndarray:
    """
    1) Finds the largest contour in 'mask' (0/255).
    2) Uses cv2.fitEllipse to fit an ellipse around that contour.
    3) Draws a 1-pixel-wide ellipse in white (255) on a black background of the same shape.

    'shrink_factor' (optional) can be < 1 to shrink the ellipse radii slightly,
    or > 1 to expand it. 1.0 => no change.
    """
    # Convert mask to 0/1 for OpenCV
    bw = (mask > 0).astype(np.uint8)

    # Find contours
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8)

    # Pick largest contour by area
    contours.sort(key=cv2.contourArea, reverse=True)
    big_contour = contours[0]

    # If the contour is too small or has fewer than 5 points, we can't fit an ellipse
    if len(big_contour) < 5:
        return np.zeros_like(mask, dtype=np.uint8)

    # Fit ellipse (center (x, y), (major, minor), angle)
    #   angle is the rotation in degrees from the x-axis
    ellipse = cv2.fitEllipse(big_contour)
    (x_center, y_center), (major, minor), angle_deg = ellipse

    # Optionally shrink or expand the ellipse
    #   e.g. major *= 0.9 => ellipse is 90% as big as normal
    major *= shrink_factor
    minor *= shrink_factor

    # Create an empty black image for the outline
    out = np.zeros_like(mask, dtype=np.uint8)

    # Draw the ellipse perimeter in white
    # thickness=1 => outline only, if you want a filled ellipse use -1
    center = (int(round(x_center)), int(round(y_center)))
    axes = (int(round(major/2)), int(round(minor/2)))  # OpenCV wants half-lengths
    angle = angle_deg  # degrees
    cv2.ellipse(out, center, axes, angle, 0, 360, 255, thickness=1)

    return out


def fit_ellipse_outline_around_mask(mask: np.ndarray) -> np.ndarray:
    """
    Given a 2D binary mask (values 0 or 255) containing some 'ellipse-like' shape,
    fit a best ellipse using image moments (regionprops), then draw **only the outline**
    (perimeter) of that fitted ellipse. The outline is a single pixel wide.

    We also shrink the resulting ellipse by a factor so it better approximates an
    "inside" boundary for the shape.

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
      4) Multiply major/minor axis lengths by SHRINK_FACTOR to get a smaller ellipse.
      5) Convert regionprops orientation to the format expected by skimage.draw.ellipse_perimeter
         (where orientation=0 means horizontal axis).
      6) Draw the ellipse perimeter in the output mask.
    """
    bw = (mask > 0)

    # Label and pick the largest region
    labeled = measure.label(bw)
    regions = measure.regionprops(labeled)
    if not regions:
        # No foreground -> return empty
        return np.zeros_like(mask, dtype=np.uint8)

    # Sort by area, pick largest region
    regions.sort(key=lambda r: r.area, reverse=True)
    region = regions[0]

    y0, x0 = region.centroid
    orientation = region.orientation  # angle from vertical in [-pi/2, pi/2]
    major_len = region.major_axis_length
    minor_len = region.minor_axis_length

    # If no significant region
    if major_len < 1 or minor_len < 1:
        return np.zeros_like(mask, dtype=np.uint8)

    # Build an empty output for the perimeter
    out_mask = np.zeros_like(mask, dtype=np.uint8)

    # Shrink the ellipse radius so it lies more inside
    r_radius = (major_len / 2.0) * SHRINK_FACTOR
    c_radius = (minor_len / 2.0) * SHRINK_FACTOR

    # regionprops orientation=0 => major axis is vertical
    # ellipse_perimeter orientation=0 => major axis is horizontal
    # => So we rotate by pi/2
    ellipse_orientation = np.pi/2 - orientation

    # Compute perimeter coords, rounding the center & radii
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
def fit_ellipse():
    in_path = "./data/ellipse_infer.png"
    out_path = "./data/ellipse_fitted.png"

    # Load the mask
    orig_mask = np.array(Image.open(in_path).convert("L"))

    # Fit the ellipse
    #outline_mask = fit_ellipse_outline_around_mask(orig_mask)
    outline_mask = fit_ellipse_contour(orig_mask, shrink_factor=SHRINK_FACTOR)

    # Save
    Image.fromarray(outline_mask).save(out_path)
    print(f"  => Saved fitted ellipse outline to {out_path}")

if __name__ == "__main__":
    fit_ellipse()


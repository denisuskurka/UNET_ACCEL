#!/usr/bin/env python

import numpy as np
from PIL import Image

# scikit-image for regionprops and drawing
from skimage import measure, draw

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
def fit_ellipse():
        in_path = "./data/ellipse_infer.png"
        orig_mask = np.array(Image.open(in_path).convert("L"))

        # Fit the perimeter
        outline_mask = fit_ellipse_outline_around_mask(orig_mask)

        # Save
        out_path = "./data/ellipse_fitted.png"
        Image.fromarray(outline_mask).save(out_path)
        print(f"  => Saved fitted ellipse outline to {out_path}")

if __name__ == "__main__":
    fit_ellipse()

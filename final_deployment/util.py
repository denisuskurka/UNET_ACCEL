# File: final_deployment/util.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import math

# Example margins: (top, bottom, left, right)
CROP_MARGIN = (
    math.floor(80),  # top
    math.floor(20),  # bottom
    0,                    # left
    math.floor(185)  # right
)

def crop_image(image, margin=CROP_MARGIN):
    """
    Crops the image by 'margin' on each side. 'margin' can be:
      1) An integer, which means crop that many pixels from every side.
      2) A 4-tuple of the form (top, bottom, left, right).

    For Pillow images, the crop box is (left, upper, right, lower).
    We convert (top, bottom, left, right) into Pillow’s (left, upper, right, lower).

    Returns the cropped image. If the margins are too large (width or height <= 0),
    it returns the original image.
    """
    # Pillow’s Image.size -> (width, height)
    w, h = image.size

    # Parse the margin parameter
    if isinstance(margin, int):
        top = bottom = left = right = margin
    elif isinstance(margin, (tuple, list)) and len(margin) == 4:
        top, bottom, left, right = margin
    else:
        raise ValueError(
            "margin must be either an integer or a 4-tuple (top, bottom, left, right)."
        )

    # Calculate new edges for Pillow crop
    new_left = left
    new_right = w - right
    new_top = top
    new_bottom = h - bottom

    # Check for valid cropping range
    if new_left >= new_right or new_top >= new_bottom:
        print(
            f"Warning: cannot crop with margin {margin} for a {w}x{h} image. "
            "Returning original."
        )
        return image

    # Pillow crop expects (left, upper, right, lower)
    return image.crop((new_left, new_top, new_right, new_bottom))

# File: final_deployment/paste_ellipse_back.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
from PIL import Image

def paste_ellipse_back(
    big_image_path: str,
    small_painted_path: str,
    out_path: str,
    bbox: tuple
):
    """
    Pastes `small_painted_path` into `big_image_path` at bounding-box offset.
      - big_image_path:  e.g. './data/cropped.png' (the larger cropped image)
      - small_painted_path: e.g. './data/painted_ellipse_final.png' 
          (the bounding-box region, same size as (xmax-xmin, ymax-ymin),
           with your ellipse lines drawn)
      - out_path: e.g. './data/painted_in_cropped.png'
      - bbox: (xmin, ymin, xmax, ymax)
    """
    (xmin, ymin, xmax, ymax) = bbox
    
    # 1) Load bigger image
    big_img = Image.open(big_image_path).convert("RGB")
    
    # 2) Load the smaller region that already has the ellipse drawn
    small_img = Image.open(small_painted_path).convert("RGB")
    
    # 3) Check sizes
    w_small, h_small = small_img.size
    if (xmax - xmin) != w_small or (ymax - ymin) != h_small:
        print(f"WARNING: bounding box size ({xmax - xmin}, {ymax - ymin}) "
              f"!= painted image size ({w_small}, {h_small}).")
    
    # 4) Paste
    #    If you want partial transparency or lines only, you can pass a mask=...
    #    For direct overwrite, just do:
    big_img.paste(small_img, (xmin, ymin))
    
    # 5) Save the result
    big_img.save(out_path)
    print(f"Pasted ellipse region onto '{big_image_path}'. Saved to '{out_path}'.")


# Example usage
if __name__ == "__main__":
    # Suppose our bounding box is (x=50, y=40, x=90, y=80) in cropped.png coords
    bbox = (50, 40, 90, 80)

    paste_ellipse_back(
        big_image_path="./data/cropped.png",
        small_painted_path="./data/painted_ellipse_final.png",
        out_path="./data/painted_in_cropped.png",
        bbox=bbox
    )

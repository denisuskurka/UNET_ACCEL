import os

# File: Change this to set how many pixels are cropped from each side (top, bottom, left, right)
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import cv2
import numpy as np
import shutil

# Change this to set how many pixels are cropped from each side (top, bottom, left, right)
CROP_MARGIN = 110

TOLERANCE = 70

def crop_image(image, margin):
    """
    Crops 'margin' pixels from each side of the image.
    If the image is too small to crop this amount, it returns the original image.
    """
    h, w = image.shape[:2]
    
    # Check if there's enough room to crop
    if margin * 2 >= w or margin * 2 >= h:
        print(f"Warning: cannot crop {margin}px from each side of a {w}x{h} image. Returning original.")
        return image
    
    return image[margin:h-margin, margin:w-margin]

def extract_specific_color(
    image_path,
    color_bgr,       # (B, G, R) tuple
    tolerance=20, 
    output_path=None,
    show_result=False
):
    """
    Extracts a binary (black & white) mask for a specific color (with some tolerance).
    The mask is single-channel:
        - 255 where the color is detected
        - 0 elsewhere
    Saves the result as a PNG file if output_path is provided.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image at path: {image_path}")
        return
    
    # --- Crop the image before processing ---
    image = crop_image(image, CROP_MARGIN)
    
    # Target color
    target_b, target_g, target_r = color_bgr
    
    # Define lower/upper bounds in BGR (with tolerance)
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
    
    # Generate the binary mask
    mask = cv2.inRange(image, lower_bound, upper_bound)
    # mask is single-channel: 255 where color is in range, 0 otherwise
    
    if show_result:
        cv2.imshow("Original Image (cropped)", image)
        cv2.imshow("Mask (Single Channel)", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, mask)


def process_images_in_folder(
    folder_path,
    color_bgr,
    tolerance=20,
    output_folder=None,
    color_suffix="",   
    show_result=False
):
    """
    For all images in folder_path, creates single-channel mask files in output_folder,
    naming them {originalName}_{color_suffix}.png, and cropping them first.
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(folder_path, file_name)
            
            if output_folder:
                # Always save output as PNG, ignoring original extension
                root, _ = os.path.splitext(file_name)
                output_file_name = f"{root}_{color_suffix}.png"
                output_path = os.path.join(output_folder, output_file_name)
            else:
                output_path = None
            
            extract_specific_color(
                image_path=input_path,
                color_bgr=color_bgr,
                tolerance=tolerance,
                output_path=output_path,
                show_result=show_result
            )
            print(f"Processed mask: {input_path}")


def convert_and_copy_images_01_to_99_as_png(source_folder, destination_folder):
    """
    Converts images 01.jpg -> 01.png, 02.jpg -> 02.png, ... 99.jpg -> 99.png
    from source_folder to destination_folder, cropping them first.
    """
    os.makedirs(destination_folder, exist_ok=True)

    for i in range(1, 100):
        file_name_jpg = f"{i:02d}.jpg"    # e.g., "01.jpg"
        src_path = os.path.join(source_folder, file_name_jpg)
        
        dst_file_name = f"{i:02d}.png"   # e.g., "01.png"
        dst_path = os.path.join(destination_folder, dst_file_name)

        if os.path.isfile(src_path):
            img = cv2.imread(src_path)
            if img is not None:
                # --- Crop the original image before writing ---
                img = crop_image(img, CROP_MARGIN)
                
                cv2.imwrite(dst_path, img)
                print(f"Created image: {dst_path}")
            else:
                print(f"Failed to read: {src_path}")
        else:
            print(f"File not found: {src_path}")


def replicate_original_images_for_each_mask(masks_folder, images_folder):
    """
    For every mask file named e.g. "01_blue.png" in masks_folder:
      - Identify the prefix (e.g., "01")
      - Find the corresponding base image "01.png" in images_folder
      - Copy that base image to e.g. "01_blue.png" in images_folder
         => so it matches the mask name exactly
    Result: 
      /images/01.png (original, cropped)
      /images/01_blue.png (duplicate of 01.png, also cropped)
    so you can pair them with 
      /masks/01_blue.png (the mask)
    """
    if not os.path.exists(images_folder):
        os.makedirs(images_folder, exist_ok=True)
    
    for mask_file in os.listdir(masks_folder):
        if mask_file.lower().endswith('.png'):
            # e.g. "01_blue.png"
            mask_root, _ = os.path.splitext(mask_file)  # "01_blue"
            # The prefix is everything up to the underscore, e.g. "01"
            if "_" in mask_root:
                prefix = mask_root.split("_")[0]  # "01"
                
                # The base image is "01.png" in /images
                original_image_name = f"{prefix}.png"
                original_image_path = os.path.join(images_folder, original_image_name)
                
                if os.path.isfile(original_image_path):
                    # We'll copy that to /images/01_blue.png
                    new_image_path = os.path.join(images_folder, mask_file)
                    
                    shutil.copyfile(original_image_path, new_image_path)
                    print(f"Duplicated image as {new_image_path} to match mask {mask_file}")
                else:
                    print(f"No base image found for {mask_file}: {original_image_path}")
            else:
                # If there's no underscore, skip or handle differently
                print(f"Skipping mask with no underscore: {mask_file}")


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1) Create single-channel color masks -> /masks
    # -------------------------------------------------------------
    common_masks_folder = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/data/masks"
    
    # Folders with separate colors
    folder_path_blue = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/SN/100/Modra"
    folder_path_green = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/SN/100/Zelena"
    folder_path_red = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/SN/100/cervena"
    folder_path_pink = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/SN/100/pink"

    # BGR values for the colors
    color_bgr_blue = (255, 8, 0)
    color_bgr_green = (0, 255, 0)
    color_bgr_red = (0, 0, 255)
    color_bgr_pink = (255, 127, 236)

    # Blue masks
    #process_images_in_folder(
    #    folder_path=folder_path_blue,
    #    color_bgr=color_bgr_blue,
    #    tolerance=TOLERANCE,
    #    output_folder=common_masks_folder,
    #    color_suffix="blue",
    #    show_result=False
    #)
    
    # Green masks
    #process_images_in_folder(
    #    folder_path=folder_path_green,
    #    color_bgr=color_bgr_green,
    #    tolerance=TOLERANCE,
    #    output_folder=common_masks_folder,
    #    color_suffix="green",
    #    show_result=False
    #)

    # Red masks
    #process_images_in_folder(
    #    folder_path=folder_path_red,
    #    color_bgr=color_bgr_red,
    #    tolerance=TOLERANCE,
    #    output_folder=common_masks_folder,
    #    color_suffix="red",
    #    show_result=False
    #)

    process_images_in_folder(
        folder_path=folder_path_pink,
        color_bgr=color_bgr_pink,
        tolerance=TOLERANCE,
        output_folder=common_masks_folder,
        color_suffix="pink",
        show_result=False
    )

    print("All masking operations have completed.")

    # -------------------------------------------------------------
    # 2) Convert images [01.jpg..99.jpg] -> PNG -> /images
    # -------------------------------------------------------------
    source_folder_main = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/SN/100"
    common_images_folder = "/home/komaro/繝・せ繧ｯ繝医ャ繝・Cermak/FZ5-UNET/model_construct/data/images"

    convert_and_copy_images_01_to_99_as_png(source_folder_main, common_images_folder)
    print("Finished copying/converting 01.jpg..99.jpg as PNG (cropped).")

    # -------------------------------------------------------------
    # 3) For each mask (e.g. 01_blue.png), replicate base image 
    #    as 01_blue.png in /images
    # -------------------------------------------------------------
    replicate_original_images_for_each_mask(
        masks_folder=common_masks_folder, 
        images_folder=common_images_folder
    )
    print("Duplicated images for each mask name in /images.")


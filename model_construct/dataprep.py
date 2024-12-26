import os
import cv2
import numpy as np
import shutil

TOLERANCE = 70

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
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image at path: {image_path}")
        return
    
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
    
    # Optional: show original + the mask
    if show_result:
        cv2.imshow("Original Image", image)
        cv2.imshow("Mask (Single Channel)", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the mask if output_path is specified
    if output_path:
        # Save as PNG (single-channel)
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
    Process all images in a given folder to extract a binary mask for a specific color,
    saving them as PNG with a color-specific suffix.
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(folder_path, file_name)
            
            if output_folder:
                # Always save output as PNG, ignoring original extension
                root, _ = os.path.splitext(file_name)
                output_file_name = f"{root}_{color_suffix}.png"  # force PNG
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
            print(f"Processed: {input_path}")


def convert_and_copy_images_01_to_99_as_png(source_folder, destination_folder):
    """
    Reads images named 01.jpg, 02.jpg, ..., 99.jpg from source_folder,
    and writes them out as PNG to destination_folder (e.g. 01.png, 02.png, etc.).
    """
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for i in range(1, 100):
        # Construct filename like "01.jpg", "02.jpg", ..., "99.jpg"
        file_name_jpg = f"{i:02d}.jpg"
        src_path = os.path.join(source_folder, file_name_jpg)
        
        # Output name is "01.png", "02.png", etc.
        dst_file_name = f"{i:02d}.png"
        dst_path = os.path.join(destination_folder, dst_file_name)

        if os.path.isfile(src_path):
            # Read the source image
            img = cv2.imread(src_path)
            if img is not None:
                # Write out as PNG
                cv2.imwrite(dst_path, img)
                print(f"Converted and copied: {src_path} -> {dst_path}")
            else:
                print(f"Failed to read: {src_path}")
        else:
            print(f"File not found: {src_path}")


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1) Extract color masks (blue, green, red) -> single-channel PNG
    # -------------------------------------------------------------
    common_output_folder = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/data/masks"
    
    # -------------- Blue --------------
    folder_path_blue = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/SN/100/Modra"
    color_bgr_blue = (255, 8, 0)
    process_images_in_folder(
        folder_path=folder_path_blue,
        color_bgr=color_bgr_blue,
        tolerance=TOLERANCE,
        output_folder=common_output_folder,
        color_suffix="blue",
        show_result=False
    )
    
    # -------------- Green --------------
    folder_path_green = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/SN/100/Zelena"
    color_bgr_green = (0, 255, 0)
    process_images_in_folder(
        folder_path=folder_path_green,
        color_bgr=color_bgr_green,
        tolerance=TOLERANCE,
        output_folder=common_output_folder,
        color_suffix="green",
        show_result=False
    )

    # -------------- Red --------------
    folder_path_red = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/SN/100/cervena"
    color_bgr_red = (0, 0, 255)
    process_images_in_folder(
        folder_path=folder_path_red,
        color_bgr=color_bgr_red,
        tolerance=TOLERANCE,
        output_folder=common_output_folder,
        color_suffix="red",
        show_result=False
    )

    print("All masking operations have completed.")

    # -------------------------------------------------------------
    # 2) Copy (convert) images 01.jpg to 99.jpg -> PNG
    # -------------------------------------------------------------
    source_folder_main = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/SN/100"
    destination_folder_main = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/data/images"

    convert_and_copy_images_01_to_99_as_png(source_folder_main, destination_folder_main)
    print("Finished copying and converting 01.jpg to 99.jpg as PNG.")

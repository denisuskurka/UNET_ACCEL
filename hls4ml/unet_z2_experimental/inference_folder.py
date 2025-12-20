#!/usr/bin/env python
# File: hls4ml/unet_z2_experimental/inference_folder.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import numpy as np
import tensorflow as tf

from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from PIL import Image  # for saving PNG masks

# ------------------------------------------------------------------------------------
# CONFIGURABLE CONSTANTS
# ------------------------------------------------------------------------------------
IMAGE_HEIGHT  = 128
IMAGE_WIDTH   = 128

IMAGES_DIR           = "./data/images"
MODEL_PATH           = "best_model.h5"
PREDICTION_MASKS_DIR = "./prediction_masks"

# Toggle these to enable/disable saving of certain files
SAVE_INPUT_NPY   = False
SAVE_INPUT_RAW   = False
SAVE_PRED_NPY    = False

# Optional suffixes for the files:
INPUT_NPY_SUFFIX  = "_input.npy"
RAW_BIN_SUFFIX    = "_input.bin"
OUTPUT_NPY_SUFFIX = "_pred.npy"
# ------------------------------------------------------------------------------------

# --- Utility for “fixed-point” style encoding (optional) ---
def encode_to_int32(x):
    """
    Example: scale by 2^24 and round, storing as int32.
    """
    return np.int32(np.round(x * (2**24)))

def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    Load image from disk, decode as grayscale, scale to [0,1], and resize to (height,width).
    Returns a float32 tensor of shape (height, width, 1).
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)  # => [0,1]
    image = tf.image.resize(image, [height, width])
    return image

def save_prediction_as_png(prediction, png_path):
    """
    Convert the prediction (float array) to a grayscale [0..255] image and save as PNG.
    Uses min-max normalization across the entire mask to stretch pixel values.
    """
    # Remove any trailing channel dimension => (H, W) array
    mask_2d = np.squeeze(prediction)

    min_val = mask_2d.min()
    max_val = mask_2d.max()
    if abs(max_val - min_val) < 1e-12:
        # If the mask is constant, just zero out
        scaled = np.zeros_like(mask_2d, dtype=np.float32)
    else:
        # Min-max normalize to 0..1
        scaled = (mask_2d - min_val) / (max_val - min_val)

    # Convert to uint8 0..255
    scaled_8bit = (scaled * 255).astype(np.uint8)

    # Save using Pillow
    img = Image.fromarray(scaled_8bit, mode='L')  # 'L' means 8-bit grayscale
    img.save(png_path)
    # No display, just file save.

def main():
    # Gather all valid images in IMAGES_DIR
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(valid_exts)
    ])
    if not image_files:
        print(f"No valid images found in '{IMAGES_DIR}'. Nothing to do.")
        return

    # Create the output folder for predicted PNGs
    if not os.path.exists(PREDICTION_MASKS_DIR):
        os.makedirs(PREDICTION_MASKS_DIR)

    # Prepare custom objects if using QKeras/Pruning
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    custom_objects['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print("Loaded model from:", MODEL_PATH)

    # Process each image
    for idx, fname in enumerate(image_files, start=1):
        image_path = os.path.join(IMAGES_DIR, fname)
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path}")

        # Preprocess => (H, W, 1)
        image_tensor = load_and_preprocess_image(image_path)
        image_arr = image_tensor.numpy()  # for optional saving

        # 1) Optionally save preprocessed image as NPY
        if SAVE_INPUT_NPY:
            npy_name = os.path.splitext(fname)[0] + INPUT_NPY_SUFFIX
            np.save(npy_name, image_arr)
            print(f" - Saved preprocessed input to '{npy_name}' (shape={image_arr.shape}).")

        # 2) Optionally save raw binary (int32) representation
        if SAVE_INPUT_RAW:
            bin_name = os.path.splitext(fname)[0] + RAW_BIN_SUFFIX
            encoded_int32 = encode_to_int32(image_arr)  # shape=(H,W,1)
            encoded_int32.tofile(bin_name)
            print(f" - Saved raw input to '{bin_name}' (size={encoded_int32.size*4} bytes).")

        # 3) Run inference => shape (1, H, W, 1) or (1, H, W) depending on model
        pred = model.predict(tf.expand_dims(image_tensor, axis=0), verbose=0)
        # Squeeze out the batch dimension => shape (H, W, 1) or (H, W)
        pred_mask = np.squeeze(pred, axis=0)

        # 4) Always save the predicted mask as a PNG to PREDICTION_MASKS_DIR
        out_png_name = os.path.splitext(fname)[0] + ".png"
        out_png_path = os.path.join(PREDICTION_MASKS_DIR, out_png_name)
        save_prediction_as_png(pred_mask, out_png_path)
        print(f" - Saved prediction mask PNG -> '{out_png_path}'.")

        # 5) Optionally save the predicted mask array as .npy
        if SAVE_PRED_NPY:
            pred_npy_name = os.path.splitext(fname)[0] + OUTPUT_NPY_SUFFIX
            np.save(pred_npy_name, pred_mask)
            print(f" - Saved prediction array -> '{pred_npy_name}'.")

    print("\nAll images processed successfully.")

if __name__ == "__main__":
    main()

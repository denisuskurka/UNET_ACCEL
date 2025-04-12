#!/usr/bin/env python
"""
A simple inference script for a QKeras model.
This script:
  • Loads one image from the images folder.
  • Preprocesses it (grayscale, resized to 128×128).
  • Exports the preprocessed image as 'X_test.npy'.
  • Loads the quantized model (using a custom object scope to register QKeras layers)
    with compile=False (to avoid loading the custom loss).
  • Runs inference to get a baseline response.
  • Exports the predicted mask as 'Y_baseline.npy'.
  • Saves the predicted mask as a BMP ('ellipse_infer.bmp').
  • Displays the input image and predicted mask using Matplotlib.
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image  # for saving BMP

# Import the needed QKeras layers for custom objects.
from qkeras import QConv2DBatchnorm, QActivation

# Force TensorFlow to use the CPU only
#tf.config.set_visible_devices([], 'CPU')
print("Running on CPU only.")

# ----------------------------
# Parameters
# ----------------------------
IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128
MODEL_PATH   = "ellipse_model.h5"           # path to your saved QKeras model
INPUT_PIC    = "./data/data_cropped_final.png"
OUTPUT_EXPORT = "./data/ellipse_infer.png"

# ----------------------------
# Utility Functions
# ----------------------------
def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    Loads an image file, decodes it as a grayscale image,
    converts it to float32 in [0, 1] range, and resizes it.
    """
    image = tf.io.read_file(image_path)
    # Decode as BMP/PNG with 1 channel (grayscale)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0..1]
    image = tf.image.resize(image, [height, width])
    return image

def infer_ellipse():
    # 1) Load and preprocess the image
    image_path = INPUT_PIC
    image = load_and_preprocess_image(image_path)
    # shape: (128, 128, 1)
    image_batch = tf.expand_dims(image, axis=0)  # shape: (1, 128, 128, 1)
    print(f"Input image shape (model input): {image_batch.shape}")

    # Save the preprocessed image as X_test.npy
    np.save("X_test.npy", image_batch.numpy())
    print("Saved preprocessed input to 'X_test.npy'.")

    # 2) Load the QKeras model
    custom_objects = {
        "QConv2DBatchnorm": QConv2DBatchnorm,
        "QActivation": QActivation,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded from:", MODEL_PATH)

    # 3) Run inference
    print("Running inference...")
    pred = model.predict(image_batch)
    # Remove batch dimension => shape: (128, 128, 1) or (128, 128) if last layer is just a single channel
    pred_mask = np.squeeze(pred)
    print("Prediction shape:", pred_mask.shape)

    # 4) Export the raw predicted mask as 'Y_baseline.npy'
    np.save("Y_baseline.npy", pred_mask)
    print("Saved predicted mask to 'Y_baseline.npy'.")

    # 5) Convert the mask to [0..255] and save as BMP
    #    Typically, you might have values in [0..1] if your last layer is a sigmoid
    #    or any other range. Adjust as needed.
    pred_mask_clamped = np.clip(pred_mask, 0.0, 1.0)
    pred_mask_255 = (pred_mask_clamped * 255).astype(np.uint8)

    # PIL expects 2D (H, W) for mode='L', or (H, W, 3) for RGB, etc.
    pred_img = Image.fromarray(pred_mask_255, mode='L')
    pred_img.save(OUTPUT_EXPORT)
    print(f"Saved predicted mask as BMP to '{OUTPUT_EXPORT}'.")

if __name__ == "__main__":
    infer_ellipse()

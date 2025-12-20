#!/usr/bin/env python
# File: hls4ml/ellipse_runet/inference.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

# Force TensorFlow to use CPU (optional)
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# Parameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128
IMAGES_DIR   = "./data/images"

MODEL_PATH    = "ellipse_regresor.h5"
INPUT_EXPORT  = "X_test.npy"
RAW_EXPORT    = "X_test.bin"
OUTPUT_EXPORT = "Y_baseline.npy"

def encode(xi):
    return np.int32(round(xi * 2**24))
encode_v = np.vectorize(encode)

def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

def get_image_x_path(images_dir, x=1, valid_exts=('.png', '.jpg', '.jpeg')):
    images = sorted(
        [fname for fname in os.listdir(images_dir) if fname.lower().endswith(valid_exts)]
    )
    if 1 <= x <= len(images):
        return os.path.join(images_dir, images[x - 1])
    return None

def draw_ellipse_on_image(image_np, params, color=(0, 255, 0)):
    img = (image_np * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cx, cy, ax1, ax2, angle = params
    center = (int(cx), int(cy))
    axes = (int(ax1), int(ax2))
    angle = float(angle)

    cv2.ellipse(img, center, axes, angle, 0, 360, color, 2)
    return img

def show_prediction_overlay(image_np, ellipse_img):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np.squeeze(), cmap='gray')
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(ellipse_img)
    plt.title("Predicted Ellipse Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    image_path = get_image_x_path(IMAGES_DIR, 3)
    if image_path is None:
        print(f"No image files found in {IMAGES_DIR}.")
        return
    print("Using image:", image_path)

    image = load_and_preprocess_image(image_path)
    image1 = np.ascontiguousarray(image)
    image1.tofile("X_test1.bin")
    image_batch = tf.expand_dims(image, axis=0)

    # Save input
    np.save(INPUT_EXPORT, image.numpy())
    image_raw = image.numpy().astype(np.float32)
    image_fixed = encode_v(image_raw)
    image_fixed.tofile(RAW_EXPORT)
    print(f"Exported input image to '{INPUT_EXPORT}' and raw bytes to '{RAW_EXPORT}'.")

    # Load model
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    custom_objects['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=custom_objects, compile=False
    )
    print("Model loaded from:", MODEL_PATH)

    # Predict ellipse parameters
    print("Running inference...")
    pred = model.predict(image_batch)
    pred_params = np.squeeze(pred)
    print("Predicted ellipse params:", pred_params)
    np.save(OUTPUT_EXPORT, pred_params)

    # Draw ellipse on image
    image_np = image.numpy().squeeze()
    ellipse_img = draw_ellipse_on_image(image_np, pred_params)

    # Show result
    show_prediction_overlay(image_np, ellipse_img)

if __name__ == "__main__":
    main()

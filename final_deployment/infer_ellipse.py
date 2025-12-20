#!/usr/bin/env python
# File: final_deployment/infer_ellipse.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
Inference script for ellipse regression model with output upscaled to original resolution.
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from qkeras import QConv2DBatchnorm, QActivation

# ----------------------------
# Parameters
# ----------------------------
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
MODEL_PATH = "ellipse_regresor.h5"
INPUT_PIC = "./data/data_cropped_final.png"
OUTPUT_EXPORT = "./data/ellipse_infer.png"

ellipse_model = None


def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    original_shape = tf.shape(image)[:2]
    image_resized = tf.image.resize(image, [height, width])
    return image, image_resized, original_shape


def draw_ellipse_on_image(image_np, params, color=(0, 255, 0)):
    img_uint8 = (image_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    cx, cy, axis1, axis2, angle = params
    center = (int(cx), int(cy))
    axes = (int(axis1), int(axis2))
    angle = float(angle)

    cv2.ellipse(img_bgr, center, axes, angle, 0, 360, color, 2)
    return img_bgr


def load_ellipse_model():
    global ellipse_model
    custom_objects = {
        "QConv2DBatchnorm": QConv2DBatchnorm,
        "QActivation": QActivation,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded from:", MODEL_PATH)
    ellipse_model = model


def infer_ellipse():
    global ellipse_model

    # Load original + preprocessed image
    original_image, image_resized, original_shape = load_and_preprocess_image(INPUT_PIC)
    original_image_np = original_image.numpy().squeeze()
    image_batch = tf.expand_dims(image_resized, axis=0)
    print(f"Original image shape: {original_image_np.shape}")
    print(f"Model input shape: {image_batch.shape}")

    np.save("X_test.npy", image_batch.numpy())

    if ellipse_model is None:
        load_ellipse_model()

    # Predict ellipse parameters
    pred = ellipse_model.predict(image_batch)
    ellipse_params = np.squeeze(pred)
    print("Predicted ellipse params:", ellipse_params)
    np.save("Y_baseline.npy", ellipse_params)

    # Draw ellipse on resized image
    image_resized_np = image_resized.numpy().squeeze()
    image_with_ellipse = draw_ellipse_on_image(image_resized_np, ellipse_params)

    # Upscale to original resolution
    output_upscaled = cv2.resize(
        image_with_ellipse,
        (int(original_shape[1]), int(original_shape[0])),
        interpolation=cv2.INTER_LINEAR
    )

    # Save as PNG
    image_rgb = cv2.cvtColor(output_upscaled, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(OUTPUT_EXPORT)
    print(f"Saved output image with ellipse to '{OUTPUT_EXPORT}'.")


if __name__ == "__main__":
    infer_ellipse()

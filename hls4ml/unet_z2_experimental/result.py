#!/usr/bin/env python
# File: hls4ml/unet_z2_experimental/result.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
This script loads the hardware output stored in 'y_hw.npy',
analyzes its shape, reshapes it into an image (assuming a 128×128 output),
and then displays the resulting mask using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128
OUTPUT_FILENAME = 'y_hw.npy'

# ----------------------------
# Load and Analyze y_hw.npy
# ----------------------------
# Load the hardware model output.
y_hw = np.load(OUTPUT_FILENAME)
print("Original y_hw shape:", y_hw.shape)

# ----------------------------
# Reshape if Necessary
# ----------------------------
# We assume that if the output is flattened (size 16384), it should be reshaped to (128,128).
if y_hw.ndim == 1 and y_hw.size == IMAGE_HEIGHT * IMAGE_WIDTH:
    y_hw_image = y_hw.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
elif y_hw.ndim == 2:
    # If it's (1,16384) or (N,16384), we take the first sample.
    if y_hw.shape[1] == IMAGE_HEIGHT * IMAGE_WIDTH:
        y_hw_image = y_hw[0].reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    else:
        y_hw_image = y_hw  # Assume it is already an image.
elif y_hw.ndim == 3:
    # If it has a channel dimension (e.g. (128,128,1)), squeeze it.
    y_hw_image = np.squeeze(y_hw)
else:
    y_hw_image = y_hw  # Use as is.

print("Reshaped y_hw image shape:", y_hw_image.shape)

# ----------------------------
# Optional Normalization
# ----------------------------
# Check the min and max values; if values exceed 1.0, normalize for display.
print("Min value:", y_hw_image.min(), "Max value:", y_hw_image.max())
if y_hw_image.max() > 1.0:
    y_hw_image = y_hw_image / y_hw_image.max()

# ----------------------------
# Display the Output Image
# ----------------------------
plt.figure(figsize=(6,6))
plt.imshow(y_hw_image, cmap='gray')
plt.title("Hardware (y_hw.npy) Output")
plt.axis('off')
plt.tight_layout()
plt.show()

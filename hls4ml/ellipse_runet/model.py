#!/usr/bin/env python
# File: hls4ml/ellipse_runet/model.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
Simple CNN to regress ellipse parameters from grayscale ultrasound input.

Predicts 5 values: (cx, cy, axis1, axis2, angle)

Architecture:
  - 3 Conv2D + MaxPool layers
  - Flatten + Dense
  - Output Dense with 5 values (no activation)

Optimized for minimal size and FPGA friendliness.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


def build_model(HEIGHT, WIDTH):
    """
    Build a compact CNN that regresses 5 ellipse parameters from grayscale input.
    """
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='input_image')

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dense(32, activation='relu', name='fc2')(x)

    # Output: cx, cy, axis1, axis2, angle
    outputs = Dense(5, name='ellipse_params')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ellipse_regressor')
    model.summary()

    return model


if __name__ == "__main__":
    HEIGHT = 128
    WIDTH = 128
    model = build_model(HEIGHT, WIDTH)

    print("\nMODEL CHECK:")
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D', 'Dense']:
            weights = layer.get_weights()
            if weights:
                w = weights[0]
                layersize = np.prod(w.shape)
                print(f"{layer.name}: {layersize}")
                if layersize > 4096:
                    print(f"Layer {layer.name} is too large ({layersize}), are you sure you want to train?")

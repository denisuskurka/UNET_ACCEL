#!/usr/bin/env python
"""
Ellipse Regression Model (Lightweight CNN) for 256x256 Input

Input:  256x256x1 Grayscale Image
Output: Vector of size 5 -> [cx, cy, axis1, axis2, angle]

Architecture:
  - 6 Downsampling blocks (Conv2D -> BN -> ReLU -> MaxPool)
  - Global Average Pooling
  - Final Dense Output Layer
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
import numpy as np

def build_ellipse_regressor(HEIGHT=256, WIDTH=256):
    """
    Builds a regression CNN for ellipse parameters optimized for 256x256 inputs.
    """
    # ---- Input ----
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='cnn_input')

    # Helper function for a Convolutional Block
    def conv_block(x, filters, kernel_size=(3, 3)):
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False # Bias is redundant when using BatchNormalization
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x

    # -------------------------------------------------------------------------
    #                             Feature Extractor
    # -------------------------------------------------------------------------
    
    # Block 1: 256x256 -> 128x128
    # We start with 16 filters to keep the model light.
    x = conv_block(inputs, filters=16)

    # Block 2: 128x128 -> 64x64
    x = conv_block(x, filters=32)
    
    # Block 3: 64x64 -> 32x32
    x = conv_block(x, filters=64)

    # Block 4: 32x32 -> 16x16
    x = conv_block(x, filters=64)

    # Block 5: 16x16 -> 8x8
    x = conv_block(x, filters=128)

    # Block 6: 8x8 -> 4x4 (Added for 256x256 support)
    # This extra depth is crucial for the model to "see" the whole object at once.
    x = conv_block(x, filters=128)

    # -------------------------------------------------------------------------
    #                             Regressor Head
    # -------------------------------------------------------------------------
    
    # Global Average Pooling transforms (4, 4, 128) -> (128,) vector.
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Optional: Dropout to prevent overfitting if you have a small dataset
    # x = Dropout(0.5)(x)

    # Dense layer to combine features
    x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense_1')(x)
    
    # Output Layer: 5 Neurons (cx, cy, major, minor, angle)
    # LINEAR activation is standard for regression.
    outputs = Dense(5, activation='linear', name='output_params')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ellipse_regressor_256')
    
    return model


if __name__ == "__main__":
    # Settings
    HEIGHT = 256
    WIDTH = 256
    
    model = build_ellipse_regressor(HEIGHT, WIDTH)
    model.summary()

    print("\nMODEL SIZE CHECK:")
    total_params = model.count_params()
    print(f"Total Parameters: {total_params}")
    
    # Check for DPU compatibility (layers shouldn't be massive)
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D', 'Dense']:
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            if layersize > 100000:
                print(f"[WARN] Layer {layer.name} is quite large ({layersize} params).")
            else:
                print(f"[OK] Layer {layer.name}: {layersize} params")

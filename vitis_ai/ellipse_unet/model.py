#!/usr/bin/env python
# File: model.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
Standard Float U-Net (Refactored from QKeras)

This model implements a standard U-Net architecture for Vitis AI deployment:
  - Four downsampling blocks
  - A bottleneck
  - Four upsampling blocks
  - A final 1ﾃ・ convolution with sigmoid activation

Changes from QKeras version:
  - QConv2DBatchnorm replaced by Conv2D + BatchNormalization
  - QActivation replaced by standard ReLU
  - Quantization parameters removed
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1

def build_model(HEIGHT, WIDTH):
    """
    Builds a standard float U-Net model suitable for Vitis AI quantization flow.
    
    Parameters:
      HEIGHT, WIDTH: Dimensions of the input image (e.g., 128ﾃ・28)
      
    Returns:
      A Keras Model instance.
    """
    # ---- Input ----
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='cnn_input')
    
    # Helper function for a standard Conv2D + BN + ReLU block
    def conv_block(x, filters, prefix):
        # --- First Convolution Layer in Block ---
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,  # Kept True to match original, though BN usually makes bias redundant
            name=f'{prefix}_conv1'
        )(x)
        x = BatchNormalization(name=f'{prefix}_bn1')(x)
        x = Activation('relu', name=f'{prefix}_act1')(x)
        
        # --- Second Convolution Layer in Block ---
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name=f'{prefix}_conv2'
        )(x)
        x = BatchNormalization(name=f'{prefix}_bn2')(x)
        x = Activation('relu', name=f'{prefix}_act2')(x)
        return x

    # -------------------------------------------------------------------------
    #                             Downsampling path
    # -------------------------------------------------------------------------
    # Block 1
    down1 = conv_block(inputs, filters=64, prefix='down1')
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(down1)
    
    # Block 2
    down2 = conv_block(pool1, filters=128, prefix='down2')
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(down2)
    
    # Block 3
    down3 = conv_block(pool2, filters=256, prefix='down3')
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(down3)

    # Block 4
    down4 = conv_block(pool3, filters=512, prefix='down4')
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(down4)
    
    # -------------------------------------------------------------------------
    #                                 Bottleneck
    # -------------------------------------------------------------------------
    bottleneck = conv_block(pool4, filters=1024, prefix='bottleneck')
    
    # -------------------------------------------------------------------------
    #                             Upsampling path
    # -------------------------------------------------------------------------
    # Block 4 (decoder)
    up4 = UpSampling2D(size=(2, 2), name='up4')(bottleneck)
    concat4 = Concatenate(name='concat4')([up4, down4])
    up4_conv = conv_block(concat4, filters=512, prefix='up4')
    
    # Block 3 (decoder)
    up3 = UpSampling2D(size=(2, 2), name='up3')(up4_conv)
    concat3 = Concatenate(name='concat3')([up3, down3])
    up3_conv = conv_block(concat3, filters=256, prefix='up3')
    
    # Block 2 (decoder)
    up2 = UpSampling2D(size=(2, 2), name='up2')(up3_conv)
    concat2 = Concatenate(name='concat2')([up2, down2])
    up2_conv = conv_block(concat2, filters=128, prefix='up2')
    
    # Block 1 (decoder)
    up1 = UpSampling2D(size=(2, 2), name='up1')(up2_conv)
    concat1 = Concatenate(name='concat1')([up1, down1])
    up1_conv = conv_block(concat1, filters=64, prefix='up1')
    
    # -------------------------------------------------------------------------
    #                                Final Output
    # -------------------------------------------------------------------------
    # Note: No BatchNormalization or ReLU here, just Sigmoid for mask generation
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='output_conv')(up1_conv)

    model = Model(inputs=inputs, outputs=outputs, name='unet_standard_float')
    # model.summary()
    
    return model

if __name__ == "__main__":
    # Example usage:
    HEIGHT = 256
    WIDTH = 256
    model = build_model(HEIGHT, WIDTH)
    model.summary()


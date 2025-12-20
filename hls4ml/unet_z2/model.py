#!/usr/bin/env python
# File: hls4ml/unet_z2/model.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
Full U-Net with QKeras layers

This model implements a standard U-Net architecture:
  - Four downsampling blocks (each two QConv2D layers + max pool)
  - A bottleneck with two QConv2D layers
  - Four upsampling blocks (each upsample + concatenate skip + two QConv2D)
  - A final 1×1 convolution producing a 1-channel output with a sigmoid activation

All QKeras layers use:
  - kernel_quantizer="quantized_bits(6,0,alpha=1)"
  - bias_quantizer="quantized_bits(6,0,alpha=1)"
  - activation= QActivation("quantized_relu(6)")

Note: The number of filters in the convolution blocks is scaled down (16→32→64→128→256)
to be more suitable for FPGA deployment, but this is still a "full" 4-level U-Net design.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from qkeras import QConv2DBatchnorm, QActivation

def build_model(HEIGHT, WIDTH):
    """
    Builds a U-Net–like model with QKeras layers.
    
    Parameters:
      HEIGHT, WIDTH: Dimensions of the input image (e.g., 128×128)
      
    Returns:
      A Keras Model instance.
    """
    # ---- Input ----
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='cnn_input')
    
    # Helper function for a QConv2D + QActivation block
    def qconv_block(x, filters, prefix):
        x = QConv2DBatchnorm(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_quantizer="quantized_bits(6,0,alpha=1)",
            bias_quantizer="quantized_bits(6,0,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name=f'{prefix}_conv1'
        )(x)
        x = QActivation("quantized_relu(6)", name=f'{prefix}_act1')(x)
        
        x = QConv2DBatchnorm(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_quantizer="quantized_bits(6,0,alpha=1)",
            bias_quantizer="quantized_bits(6,0,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name=f'{prefix}_conv2'
        )(x)
        x = QActivation("quantized_relu(6)", name=f'{prefix}_act2')(x)
        return x

    # -------------------------------------------------------------------------
    #                             Downsampling path
    # -------------------------------------------------------------------------
    # Block 1
    down1 = qconv_block(inputs, filters=64, prefix='down1')
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(down1)
    
    # Block 2
    down2 = qconv_block(pool1, filters=128, prefix='down2')
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(down2)
    
    # Block 3
    down3 = qconv_block(pool2, filters=256, prefix='down3')
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(down3)

    # Block 4
    down4 = qconv_block(pool3, filters=512, prefix='down4')
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(down4)
    
    # -------------------------------------------------------------------------
    #                                 Bottleneck
    # -------------------------------------------------------------------------
    bottleneck = qconv_block(pool4, filters=1024, prefix='bottleneck')
    
    # -------------------------------------------------------------------------
    #                             Upsampling path
    # -------------------------------------------------------------------------
    # Block 4 (decoder)
    up4 = UpSampling2D(size=(2, 2), name='up4')(bottleneck)
    concat4 = Concatenate(name='concat4')([up4, down4])
    up4_conv = qconv_block(concat4, filters=512, prefix='up4')
    
    # Block 3 (decoder)
    up3 = UpSampling2D(size=(2, 2), name='up3')(up4_conv)
    concat3 = Concatenate(name='concat3')([up3, down3])
    up3_conv = qconv_block(concat3, filters=256, prefix='up3')
    
    # Block 2 (decoder)
    up2 = UpSampling2D(size=(2, 2), name='up2')(up3_conv)
    concat2 = Concatenate(name='concat2')([up2, down2])
    up2_conv = qconv_block(concat2, filters=128, prefix='up2')
    
    # Block 1 (decoder)
    up1 = UpSampling2D(size=(2, 2), name='up1')(up2_conv)
    concat1 = Concatenate(name='concat1')([up1, down1])
    up1_conv = qconv_block(concat1, filters=64, prefix='up1')
    
    # -------------------------------------------------------------------------
    #                                Final Output
    # -------------------------------------------------------------------------
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='output_conv')(up1_conv)

    model = Model(inputs=inputs, outputs=outputs, name='unet_light')
    model.summary()
    
    return model

if __name__ == "__main__":
    # Example usage:
    HEIGHT = 128
    WIDTH = 128
    model = build_model(HEIGHT, WIDTH)

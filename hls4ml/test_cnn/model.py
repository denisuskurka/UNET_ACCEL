#!/usr/bin/env python
"""
Unet-light with QKeras

This model implements a U-Net–like architecture with:
  - An encoder block: one QConv2DBatchnorm layer with 8 filters (3×3) followed by max pooling.
  - A bottleneck: one QConv2DBatchnorm layer with 16 filters (3×3).
  - A decoder block: upsampling, concatenation (skip connection) with encoder features,
    followed by one QConv2DBatchnorm layer with 8 filters (3×3).
  - A final 1×1 convolution producing a 1-channel output with a sigmoid activation.

The QKeras layers use a kernel quantizer "quantized_bits(6,0,alpha=1)" and a bias quantizer
of the same type. Regularization, initialization, and activation (quantized_relu(6)) are provided.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from qkeras import QConv2DBatchnorm, QActivation

def build_model(HEIGHT, WIDTH):
    """
    Builds a U-Net–like model (Unet-light) with QKeras layers.
    
    Parameters:
      HEIGHT, WIDTH: Dimensions of the input image (e.g., 128×128)
      
    Returns:
      A Keras Model instance.
    """
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='cnn_input')
    
    # ---- Encoder Block ----
    # Encoder: QKeras convolution block with 8 filters, 3×3 kernel, padding same.
    encoder = QConv2DBatchnorm(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='encoder_conv'
    )(inputs)
    encoder = QActivation("quantized_relu(6)", name='encoder_act')(encoder)
    # Save the encoder output for skip connection, then downsample.
    skip_connection = encoder
    encoder_pool = MaxPooling2D(pool_size=(2, 2), name='encoder_pool')(encoder)
    
    # ---- Bottleneck ----
    bottleneck = QConv2DBatchnorm(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='bottleneck_conv'
    )(encoder_pool)
    bottleneck = QActivation("quantized_relu(6)", name='bottleneck_act')(bottleneck)
    
    # ---- Decoder Block ----
    # Upsample to recover spatial dimensions.
    decoder_upsample = UpSampling2D(size=(2, 2), name='decoder_upsample')(bottleneck)
    # Concatenate skip connection from encoder.
    decoder_concat = Concatenate(name='skip_concat')([decoder_upsample, skip_connection])
    # Apply a QKeras convolution block with 8 filters.
    decoder_conv = QConv2DBatchnorm(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='decoder_conv'
    )(decoder_concat)
    decoder_act = QActivation("quantized_relu(6)", name='decoder_act')(decoder_conv)
    
    # ---- Output Layer ----
    # A 1×1 convolution to produce the final segmentation mask.
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='output_conv')(decoder_act)
    
    # Create the model.
    model = Model(inputs=inputs, outputs=outputs, name='unet_light')
    model.summary()
    return model

if __name__ == "__main__":
    # Example usage:
    HEIGHT = 128
    WIDTH = 128
    model = build_model(HEIGHT, WIDTH)

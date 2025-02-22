#!/usr/bin/env python
"""
Full U-Net with QKeras layers (reduced version)

This model implements a small U-Net architecture:
  - One downsampling block (QConv2D + max pool)
  - A bottleneck block
  - One upsampling block (upsample + skip connection + QConv2D)
  - A final 1×1 convolution producing a single-channel logit (no sigmoid).

All QKeras layers use:
  - kernel_quantizer="quantized_bits(6,0,alpha=1)"
  - bias_quantizer="quantized_bits(6,0,alpha=1)"
  - activation= QActivation("quantized_relu(6)")

Note: The number of filters in the convolution blocks is reduced for FPGA deployment.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from qkeras import QConv2DBatchnorm, QActivation, QConv2D
import numpy as np

def build_model(HEIGHT, WIDTH):
    """
    Builds a small UNet-like model with QKeras layers.
    
    Parameters:
      HEIGHT, WIDTH: Dimensions of the input image (e.g., 128×128)
      
    Returns:
      A Keras Model instance producing raw logits (no final sigmoid).
    """
    # ---- Input ----
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='cnn_input')
    
    # Helper function for a QConv2D + QActivation block
    def qconv_block(x, filters, prefix):
        x = QConv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_quantizer="quantized_bits(32,8,alpha=1)",
            bias_quantizer="quantized_bits(32,8,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0000),
            use_bias=True,
            name=f'{prefix}_conv1'
        )(x)
        x = QActivation("quantized_relu(16,8)", name=f'{prefix}_act1')(x)
        return x

    # -------------------------------------------------------------------------
    #                             Downsampling path
    # -------------------------------------------------------------------------
    # Block 1
    down1 = qconv_block(inputs, filters=4, prefix='down1')
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(down1)
    
    # -------------------------------------------------------------------------
    #                                 Bottleneck
    # -------------------------------------------------------------------------
    bottleneck = qconv_block(pool1, filters=8, prefix='bottleneck')
    
    # -------------------------------------------------------------------------
    #                             Upsampling path
    # -------------------------------------------------------------------------
    # Up block
    up4 = UpSampling2D(size=(2, 2), name='up4')(bottleneck)
    concat4 = Concatenate(name='concat4')([up4, down1])
    up4_conv = qconv_block(concat4, filters=8, prefix='up4')
    up3_conv = qconv_block(up4_conv, filters=4, prefix='up3')
    up2_conv = qconv_block(up3_conv, filters=2, prefix='up2')

    # -------------------------------------------------------------------------
    #                                Final Output
    # -------------------------------------------------------------------------
    # Produce 1 channel of logits (no sigmoid)
    outputs = QConv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(32,8,alpha=1)",
        bias_quantizer="quantized_bits(32,8,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0000),
        use_bias=True,
        activation=None,  # no activation => raw logits
        name='output_conv'
    )(up2_conv)

    model = Model(inputs=inputs, outputs=outputs, name='unet_light')
    model.summary()
    
    return model

if __name__ == "__main__":
    # Example usage:
    HEIGHT = 128
    WIDTH = 128
    model = build_model(HEIGHT, WIDTH)

    print("MODEL CHECK:")
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D', 'Dense']:
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            print("{}: {}".format(layer.name, layersize))  # 0 = weights, 1 = biases
            if layersize > 4096:
                print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name, layersize))

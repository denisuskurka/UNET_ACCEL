#!/usr/bin/env python
"""
Float (non-quantized) version of a "reduced U-Net" architecture.

This model implements a small U-Net architecture:
  - One downsampling block (Conv2D + max pool)
  - A bottleneck block
  - One upsampling block (upsample + skip connection + Conv2D)
  - A final 1×1 convolution producing a single-channel logit (no sigmoid).

No QKeras layers are used; all layers are standard float Keras operations.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Conv2D,
    ReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1

def build_model(HEIGHT, WIDTH):
    """
    Builds a small UNet-like model using standard Keras float layers.

    Parameters:
      HEIGHT, WIDTH: Dimensions of the input image (e.g., 128×128)

    Returns:
      A Keras Model instance producing raw logits (no final sigmoid).
    """
    # ---- Input ----
    input_shape = (HEIGHT, WIDTH, 1)
    inputs = Input(shape=input_shape, name='cnn_input')
    
    # Helper function for a Conv2D + ReLU block
    def conv_block(x, filters, prefix):
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',     # typical choice
            kernel_regularizer=l1(0.0),         # set to 0.0 or as needed
            use_bias=True,
            name=f'{prefix}_conv1'
        )(x)
        x = ReLU(name=f'{prefix}_act1')(x)
        return x

    # -------------------------------------------------------------------------
    #                             Downsampling path
    # -------------------------------------------------------------------------
    # Block 1
    down1 = conv_block(inputs, filters=4, prefix='down1')
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(down1)
    
    # -------------------------------------------------------------------------
    #                                 Bottleneck
    # -------------------------------------------------------------------------
    bottleneck = conv_block(pool1, filters=8, prefix='bottleneck')
    
    # -------------------------------------------------------------------------
    #                             Upsampling path
    # -------------------------------------------------------------------------
    up4 = UpSampling2D(size=(2, 2), name='up4')(bottleneck)
    concat4 = Concatenate(name='concat4')([up4, down1])
    up4_conv = conv_block(concat4, filters=8, prefix='up4')
    up3_conv = conv_block(up4_conv, filters=4, prefix='up3')
    up2_conv = conv_block(up3_conv, filters=2, prefix='up2')

    # -------------------------------------------------------------------------
    #                                Final Output
    # -------------------------------------------------------------------------
    # Produce 1 channel of logits (no activation => raw output)
    outputs = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation=None,
        name='output_conv'
    )(up2_conv)

    model = Model(inputs=inputs, outputs=outputs, name='unet_light_float')
    model.summary()
    
    return model

if __name__ == "__main__":
    # Example usage:
    HEIGHT = 128
    WIDTH = 128
    model = build_model(HEIGHT, WIDTH)

    print("MODEL CHECK:")
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            w = layer.get_weights()[0]
            layersize = w.size  # NumPy .size gives total number of elements
            print("{}: {}".format(layer.name, layersize))
            if layersize > 4096:
                print(f"Layer {layer.name} is too large ({layersize}), "
                      "are you sure you want to train?")

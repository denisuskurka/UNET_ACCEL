# File: archive/paddlepaddle-old/unet/model.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

# model.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import matplotlib.pyplot as plt

# Define image dimensions as constants
IMG_HEIGHT = 64  # Change this to the desired height
IMG_WIDTH = 64   # Change this to the desired width

class UNet(nn.Layer):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # Encoder path
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder path
        self.up4 = nn.Conv2DTranspose(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.Conv2DTranspose(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2D(64, 1, kernel_size=1)  # Output a single channel for binary segmentation

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.dec4(paddle.concat([self.up4(bottleneck), enc4], axis=1))
        dec3 = self.dec3(paddle.concat([self.up3(dec4), enc3], axis=1))
        dec2 = self.dec2(paddle.concat([self.up2(dec3), enc2], axis=1))
        dec1 = self.dec1(paddle.concat([self.up1(dec2), enc1], axis=1))

        # Output
        return self.out_conv(dec1)


# Visualize the first image and mask
def visualize_sample(image, mask):
    # Convert tensors to numpy arrays for display
    image = image.numpy().transpose(1, 2, 0)  # Change from [C, H, W] to [H, W, C] for plotting
    mask = mask.numpy()

    # Plot the image and mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask")
    ax[1].axis("off")

    plt.show()

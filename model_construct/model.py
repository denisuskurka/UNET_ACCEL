import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant

class UNetBrevitas(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetBrevitas, self).__init__()
        
        def conv_block(in_ch, out_ch):
            """
            Returns a sequential block of:
              QuantConv2d -> QuantReLU -> QuantConv2d -> QuantReLU
            """
            return nn.Sequential(
                qnn.QuantConv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    weight_quant=WeightQuant,
                    weight_bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantReLU(
                    act_quant=ActQuant,
                    bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantConv2d(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    weight_quant=WeightQuant,
                    weight_bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantReLU(
                    act_quant=ActQuant,
                    bit_width=8,
                    return_quant_tensor=True
                )
            )
        
        # U-Net Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # U-Net Decoder (using QuantConvTranspose2d)
        self.up4 = qnn.QuantConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=True
        )
        self.dec4 = conv_block(1024, 512)
        
        self.up3 = qnn.QuantConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=True
        )
        self.dec3 = conv_block(512, 256)

        self.up2 = qnn.QuantConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=True
        )
        self.dec2 = conv_block(256, 128)

        self.up1 = qnn.QuantConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=True
        )
        self.dec1 = conv_block(128, 64)

        # Final output convolution (using QuantConv2d)
        # For segmentation, typically out_channels = 1 for a single mask
        self.out_conv = qnn.QuantConv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=False  # Typically final layer returns a normal tensor
        )

    def forward(self, x):
        # The forward pass logic parallels your PaddlePaddle code
        # Down-sampling path
        enc1_out = self.enc1(x)
        enc2_in = F.max_pool2d(enc1_out, 2)
        enc2_out = self.enc2(enc2_in)
        
        enc3_in = F.max_pool2d(enc2_out, 2)
        enc3_out = self.enc3(enc3_in)

        enc4_in = F.max_pool2d(enc3_out, 2)
        enc4_out = self.enc4(enc4_in)

        # Bottleneck
        bottleneck_in = F.max_pool2d(enc4_out, 2)
        bottleneck_out = self.bottleneck(bottleneck_in)

        # Up-sampling path + skip connections
        up4_out = self.up4(bottleneck_out)
        dec4_in = torch.cat([up4_out, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_in)

        up3_out = self.up3(dec4_out)
        dec3_in = torch.cat([up3_out, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_in)

        up2_out = self.up2(dec3_out)
        dec2_in = torch.cat([up2_out, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)

        up1_out = self.up1(dec2_out)
        dec1_in = torch.cat([up1_out, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)

        out = self.out_conv(dec1_out)
        return out

if __name__ == "__main__":
    # Example input (batch_size=1, 3 channels, e.g. 416x576)
    input_tensor = torch.randn(1, 3, 416, 576)
    
    model = UNetBrevitas(in_channels=3, out_channels=1)
    with torch.no_grad():
        output = model(input_tensor)
    print("Output shape:", output.shape)
    # Should be [1, 1, 416, 576] for this U-Net

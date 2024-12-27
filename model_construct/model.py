import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFloatBatchQuant2d as ActQuantIdentity  # Signed
from brevitas.quant import Uint8ActPerTensorFloatBatchQuant2d as ActQuantReLU      # Unsigned

PRINT_PASS=True  # Set to True to enable print statements

class UNetBrevitasFINN(nn.Module):
    """
    A U-Net compatible with FINN using separate quantizers for ReLU and QuantIdentity layers.
    Skip connections are implemented via element-wise addition.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(UNetBrevitasFINN, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                qnn.QuantConv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_quant=WeightQuant,
                    weight_bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantReLU(
                    act_quant=ActQuantReLU,  # Unsigned
                    bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantConv2d(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_quant=WeightQuant,
                    weight_bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantReLU(
                    act_quant=ActQuantReLU,  # Unsigned
                    bit_width=8,
                    return_quant_tensor=True
                )
            )

        def upsample_block(in_ch, out_ch):
            return nn.Sequential(
                qnn.QuantConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                    weight_quant=WeightQuant,
                    weight_bit_width=8,
                    return_quant_tensor=True
                ),
                qnn.QuantReLU(
                    act_quant=ActQuantReLU,  # Unsigned
                    bit_width=8,
                    return_quant_tensor=True
                )
            )

        def downsample_block(in_ch, out_ch):
            return qnn.QuantConv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=8,
                return_quant_tensor=True
            )

        # QuantIdentity with signed quantizer
        self.fix_scale = qnn.QuantIdentity(
            act_quant=ActQuantIdentity,  # Signed
            bit_width=8,
            return_quant_tensor=True
        )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.down1 = downsample_block(64, 128)
        self.enc2 = conv_block(128, 128)
        self.down2 = downsample_block(128, 256)
        self.enc3 = conv_block(256, 256)
        self.down3 = downsample_block(256, 512)
        self.enc4 = conv_block(512, 512)

        # Bottleneck
        self.down4 = downsample_block(512, 1024)
        self.bottleneck = conv_block(1024, 1024)

        # Decoder
        self.up4 = upsample_block(1024, 512)
        self.dec4 = conv_block(512, 512)  # Adjusted to accept 512 channels

        self.up3 = upsample_block(512, 256)
        self.dec3 = conv_block(256, 256)  # Adjusted to accept 256 channels

        self.up2 = upsample_block(256, 128)
        self.dec2 = conv_block(128, 128)  # Adjusted to accept 128 channels

        self.up1 = upsample_block(128, 64)
        self.dec1 = conv_block(64, 64)     # Adjusted to accept 64 channels

        # Final output layer
        self.out_conv = qnn.QuantConv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=False  # Final layer does not need quantization
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        if PRINT_PASS: print(f"enc1_out shape: {enc1_out.shape}")  # Should be [1, 64, 128, 128]
        enc2_in = self.fix_scale(self.down1(enc1_out))
        enc2_out = self.enc2(enc2_in)
        if PRINT_PASS: print(f"enc2_out shape: {enc2_out.shape}")  # Should be [1, 128, 64, 64]

        enc3_in = self.fix_scale(self.down2(enc2_out))
        enc3_out = self.enc3(enc3_in)
        if PRINT_PASS: print(f"enc3_out shape: {enc3_out.shape}")  # Should be [1, 256, 32, 32]

        enc4_in = self.fix_scale(self.down3(enc3_out))
        enc4_out = self.enc4(enc4_in)
        if PRINT_PASS: print(f"enc4_out shape: {enc4_out.shape}")  # Should be [1, 512, 16, 16]

        bottleneck_in = self.fix_scale(self.down4(enc4_out))
        bottleneck_out = self.bottleneck(bottleneck_in)
        if PRINT_PASS: print(f"bottleneck_out shape: {bottleneck_out.shape}")  # Should be [1, 1024, 8, 8]

        # Decoder
        up4_out = self.up4(bottleneck_out)
        if PRINT_PASS: print(f"up4_out shape: {up4_out.shape}")  # Should be [1, 512, 16, 16]
        up4_out = self.fix_scale(up4_out)          # Align scale (signed)
        enc4_out = self.fix_scale(enc4_out)        # Align scale (signed)
        dec4_in = up4_out + enc4_out               # Skip Connection via Addition
        if PRINT_PASS: print(f"dec4_in shape: {dec4_in.shape}")    # Should be [1, 512, 16, 16]
        dec4_out = self.dec4(dec4_in)
        if PRINT_PASS: print(f"dec4_out shape: {dec4_out.shape}")  # Should be [1, 512, 16, 16]

        up3_out = self.up3(dec4_out)
        if PRINT_PASS: print(f"up3_out shape: {up3_out.shape}")    # Should be [1, 256, 32, 32]
        up3_out = self.fix_scale(up3_out)          # Align scale (signed)
        enc3_out = self.fix_scale(enc3_out)        # Align scale (signed)
        dec3_in = up3_out + enc3_out               # Skip Connection via Addition
        if PRINT_PASS: print(f"dec3_in shape: {dec3_in.shape}")    # Should be [1, 256, 32, 32]
        dec3_out = self.dec3(dec3_in)
        if PRINT_PASS: print(f"dec3_out shape: {dec3_out.shape}")  # Should be [1, 256, 32, 32]

        up2_out = self.up2(dec3_out)
        if PRINT_PASS: print(f"up2_out shape: {up2_out.shape}")    # Should be [1, 128, 64, 64]
        up2_out = self.fix_scale(up2_out)          # Align scale (signed)
        enc2_out = self.fix_scale(enc2_out)        # Align scale (signed)
        dec2_in = up2_out + enc2_out               # Skip Connection via Addition
        if PRINT_PASS: print(f"dec2_in shape: {dec2_in.shape}")    # Should be [1, 128, 64, 64]
        dec2_out = self.dec2(dec2_in)
        if PRINT_PASS: print(f"dec2_out shape: {dec2_out.shape}")  # Should be [1, 128, 64, 64]

        up1_out = self.up1(dec2_out)
        if PRINT_PASS: print(f"up1_out shape: {up1_out.shape}")    # Should be [1, 64, 128, 128]
        up1_out = self.fix_scale(up1_out)          # Align scale (signed)
        enc1_out = self.fix_scale(enc1_out)        # Align scale (signed)
        dec1_in = up1_out + enc1_out               # Skip Connection via Addition
        if PRINT_PASS: print(f"dec1_in shape: {dec1_in.shape}")    # Should be [1, 64, 128, 128]
        dec1_out = self.dec1(dec1_in)
        if PRINT_PASS: print(f"dec1_out shape: {dec1_out.shape}")  # Should be [1, 64, 128, 128]

        return self.out_conv(dec1_out)

###############################################################################
# Quick test if running standalone
###############################################################################
if __name__ == "__main__":
    # Example input: (batch_size=1, 1 channel, 128x128)
    test_input = torch.randn(1, 1, 128, 128)

    model = UNetBrevitasFINN(in_channels=1, out_channels=1)
    model.eval()

    with torch.no_grad():
        output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)
    # Expect shape (1, 1, 128, 128)

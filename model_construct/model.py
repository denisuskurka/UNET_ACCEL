import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant


###############################################################################
# A U-Net that uses nearest-neighbor upsampling + QuantConv2d
# instead of QuantConvTranspose2d.
###############################################################################
class UNetBrevitas(nn.Module):
    """
    A U-Net in Brevitas that avoids ConvTranspose2d. Instead, it:
      - Upsamples spatially by factor of 2 (nearest-neighbor)
      - Applies a quantized Conv2d to reduce channel depth.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(UNetBrevitas, self).__init__()

        #######################################################################
        # 1) Basic conv block (Conv -> ReLU -> Conv -> ReLU)
        #######################################################################
        def conv_block(in_ch, out_ch):
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

        #######################################################################
        # 2) Upsample block (nearest-neighbor upsample + quantized conv)
        #    This replaces a typical ConvTranspose2d(in_ch, out_ch, kernel=2, stride=2).
        #    We do an nn.Upsample(..., mode='nearest') followed by a QuantConv2d
        #    that maps 'in_ch' to 'out_ch'. 
        #######################################################################
        def upsample_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
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
                    act_quant=ActQuant,
                    bit_width=8,
                    return_quant_tensor=True
                )
            )

        #######################################################################
        # 3) Re-quant layer to unify scale if needed
        #######################################################################
        self.fix_scale = qnn.QuantIdentity(
            act_quant=ActQuant,
            bit_width=8,
            return_quant_tensor=True
        )

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = conv_block(512, 1024)

        # -------------------------
        # Decoder (using upsample_block)
        # -------------------------
        self.up4 = upsample_block(1024, 512)
        self.dec4 = conv_block(1024, 512)

        self.up3 = upsample_block(512, 256)
        self.dec3 = conv_block(512, 256)

        self.up2 = upsample_block(256, 128)
        self.dec2 = conv_block(256, 128)

        self.up1 = upsample_block(128, 64)
        self.dec1 = conv_block(128, 64)

        # -------------------------
        # Final output (1 channel mask)
        # -------------------------
        self.out_conv = qnn.QuantConv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=False
        )

    def forward(self, x):
        # -------------------------
        # Encoder
        # -------------------------
        enc1_out = self.enc1(x)
        enc1_out = self.fix_scale(enc1_out)
        enc2_in = F.max_pool2d(enc1_out, 2)

        enc2_out = self.enc2(enc2_in)
        enc2_out = self.fix_scale(enc2_out)
        enc3_in = F.max_pool2d(enc2_out, 2)

        enc3_out = self.enc3(enc3_in)
        enc3_out = self.fix_scale(enc3_out)
        enc4_in = F.max_pool2d(enc3_out, 2)

        enc4_out = self.enc4(enc4_in)
        enc4_out = self.fix_scale(enc4_out)

        # -------------------------
        # Bottleneck
        # -------------------------
        bottleneck_in = F.max_pool2d(enc4_out, 2)
        bottleneck_out = self.bottleneck(bottleneck_in)
        bottleneck_out = self.fix_scale(bottleneck_out)

        # -------------------------
        # Decoder level 4
        # (Upsample 1024->512, cat with enc4)
        # -------------------------
        up4_out = self.up4(bottleneck_out)   # shape (N, 512, H/8, W/8)
        up4_out = self.fix_scale(up4_out)
        enc4_out = self.fix_scale(enc4_out)
        dec4_in = torch.cat([up4_out, enc4_out], dim=1)  # (N, 1024, H/8, W/8)
        dec4_out = self.dec4(dec4_in)
        dec4_out = self.fix_scale(dec4_out)

        # -------------------------
        # Decoder level 3
        # (Upsample 512->256, cat with enc3)
        # -------------------------
        up3_out = self.up3(dec4_out)
        up3_out = self.fix_scale(up3_out)
        enc3_out = self.fix_scale(enc3_out)
        dec3_in = torch.cat([up3_out, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_in)
        dec3_out = self.fix_scale(dec3_out)

        # -------------------------
        # Decoder level 2
        # (Upsample 256->128, cat with enc2)
        # -------------------------
        up2_out = self.up2(dec3_out)
        up2_out = self.fix_scale(up2_out)
        enc2_out = self.fix_scale(enc2_out)
        dec2_in = torch.cat([up2_out, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        dec2_out = self.fix_scale(dec2_out)

        # -------------------------
        # Decoder level 1
        # (Upsample 128->64, cat with enc1)
        # -------------------------
        up1_out = self.up1(dec2_out)
        up1_out = self.fix_scale(up1_out)
        enc1_out = self.fix_scale(enc1_out)
        dec1_in = torch.cat([up1_out, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        dec1_out = self.fix_scale(dec1_out)

        # -------------------------
        # Output
        # -------------------------
        out = self.out_conv(dec1_out)
        return out


###############################################################################
# Quick test if running standalone
###############################################################################
if __name__ == "__main__":
    # Example input: (batch_size=1, 3 channels, 128x128)
    test_input = torch.randn(1, 3, 128, 128)

    model = UNetBrevitas(in_channels=3, out_channels=1)
    model.eval()

    with torch.no_grad():
        output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)
    # Expect shape (1, 1, 128, 128)

import torch

# File: Example quant modules (must exist in your codebase):
# Author: Denis Kurka
# Year: 2025
# License: CC0

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Module, ModuleList, MaxPool2d
from brevitas.nn import QuantConv2d, QuantConvTranspose2d, QuantIdentity
from brevitas.core.restrict_val import RestrictValueType

# Example quant modules (must exist in your codebase):
from common import CommonActQuant, CommonWeightQuant

DEBUG = False

ENCODER_CHANNELS = [64, 128, 256, 512]
DECODER_CHANNELS = [512, 256, 128, 64]
BOTTLE_NECK_CHANNELS = 1024
KERNEL_SIZE = 3
POOL_SIZE = 2

class UNetQ(Module):
    """
    A UNet-like network fully quantized with Brevitas, suitable for FINN hardware
    generation. Expects input shape that divides evenly with successive 2x pool,
    e.g. 128x128, 256x256, etc. to avoid float interpolation.
    """
    def __init__(self, in_channels=1, out_channels=1,
                 weight_bit_width=8, act_bit_width=8, in_bit_width=8):
        super(UNetQ, self).__init__()

        # ---------------------------------------------------------------------
        # 1) Quantize the input itself to ensure the first convolution
        #    sees integer data
        # ---------------------------------------------------------------------
        self.in_quant = QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-in_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO
        )

        # ----------------
        # Encoder modules
        # ----------------
        self.encoder = ModuleList()
        prev_channels = in_channels
        for out_ch in ENCODER_CHANNELS:
            # Convolution
            self.encoder.append(
                QuantConv2d(
                    in_channels=prev_channels,
                    out_channels=out_ch,
                    kernel_size=KERNEL_SIZE,
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            # BatchNorm
            self.encoder.append(BatchNorm2d(out_ch, eps=1e-4))
            # Activation
            self.encoder.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width,
                    min_val=0.0,
                    max_val=1.0 - 2.0 ** (-act_bit_width),
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO
                )
            )
            # MaxPool
            self.encoder.append(MaxPool2d(kernel_size=POOL_SIZE))
            prev_channels = out_ch

        # -----------
        # Bottleneck
        # -----------
        self.bottleneck = ModuleList()
        self.bottleneck.append(
            QuantConv2d(
                in_channels=prev_channels,
                out_channels=BOTTLE_NECK_CHANNELS,
                kernel_size=KERNEL_SIZE,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width
            )
        )
        self.bottleneck.append(BatchNorm2d(BOTTLE_NECK_CHANNELS, eps=1e-4))
        self.bottleneck.append(
            QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width,
                min_val=0.0,
                max_val=1.0 - 2.0 ** (-act_bit_width),
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO
            )
        )

        # -----------
        # Decoder
        # -----------
        self.decoder = ModuleList()
        current_channels = BOTTLE_NECK_CHANNELS

        for out_ch in DECODER_CHANNELS:
            # 1) Transposed Conv for upsampling
            self.decoder.append(
                QuantConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            self.decoder.append(BatchNorm2d(out_ch, eps=1e-4))
            self.decoder.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width,
                    min_val=0.0,
                    max_val=1.0 - 2.0 ** (-act_bit_width),
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO
                )
            )
            # 2) Conv after skip connection (concatenation)
            concatenated_channels = out_ch + (current_channels // 2)
            self.decoder.append(
                QuantConv2d(
                    in_channels=concatenated_channels,
                    out_channels=out_ch,
                    kernel_size=KERNEL_SIZE,
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            self.decoder.append(BatchNorm2d(out_ch, eps=1e-4))
            self.decoder.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width,
                    min_val=0.0,
                    max_val=1.0 - 2.0 ** (-act_bit_width),
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO
                )
            )
            current_channels = out_ch

        # -----------------------
        # Final output layer
        # -----------------------
        self.out_conv = QuantConv2d(
            in_channels=DECODER_CHANNELS[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )

        # Optionally init weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantConvTranspose2d)):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        # ------------------------------------------------------------------
        # 1) Quantize the input so it becomes INT8 right away
        # ------------------------------------------------------------------
        x = self.in_quant(x)

        # We'll store the output of each encoder block BEFORE the MaxPool
        # so that the transposed conv upsampling lines up exactly
        encoder_outputs = []

        # -------------------
        # Forward: Encoder
        # -------------------
        # We have groups of (Conv -> BN -> Act -> Pool)
        # so step in groups of 4 through self.encoder
        for i in range(0, len(self.encoder), 4):
            conv = self.encoder[i]
            bn = self.encoder[i+1]
            act = self.encoder[i+2]
            pool = self.encoder[i+3]

            x = conv(x)
            x = bn(x)
            x = act(x)
            # store the skip connection output here (before pooling)
            encoder_outputs.append(x)
            x = pool(x)
            if DEBUG:
                print(f"Encoder block {i//4}: {x.shape}")

        # -----------
        # Bottleneck
        # -----------
        for layer in self.bottleneck:
            x = layer(x)
        if DEBUG:
            print(f"Bottleneck output: {x.shape}")

        # -------------------
        # Forward: Decoder
        # -------------------
        # Each iteration: (TransposeConv -> BN -> Act) -> Concat -> (Conv -> BN -> Act)
        # step in groups of 6 through self.decoder
        for i in range(0, len(self.decoder), 6):
            trans_conv = self.decoder[i]
            trans_bn = self.decoder[i+1]
            trans_act = self.decoder[i+2]

            # upsample
            x = trans_conv(x)
            x = trans_bn(x)
            x = trans_act(x)

            # skip connection from the last item in encoder_outputs
            skip_idx = len(encoder_outputs) - 1 - i // 6
            skip_tensor = encoder_outputs[skip_idx]
            if DEBUG:
                print(f"Decoder block {i//6}: upsampled shape={x.shape}, skip shape={skip_tensor.shape}")

            # cat
            x = torch.cat([x, skip_tensor], dim=1)

            # conv after skip
            conv = self.decoder[i+3]
            bn = self.decoder[i+4]
            act = self.decoder[i+5]
            x = conv(x)
            x = bn(x)
            x = act(x)

            if DEBUG:
                print(f"Decoder block {i//6} output: {x.shape}")

        # --------------
        # Final Output
        # --------------
        x = self.out_conv(x)

        # NOTE: We do NOT do F.interpolate to match original input shape
        # because that would reintroduce float ops. If you need arbitrary shape
        # restoration, you'll need integer upsampling or an integer-friendly approach.
        return x


if __name__ == "__main__":
    # Simple test
    model = UNetQ(
        in_channels=1, 
        out_channels=1,
        weight_bit_width=8,
        act_bit_width=8,
        in_bit_width=8
    )
    model.eval()

    # Example input: (batch_size=1, in_channels=1, height=128, width=128)
    # This dimension (128x128) should downsample & upsample nicely by factor 2^4
    test_input = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        output = model(test_input)

    print("Input shape: ", test_input.shape)
    print("Output shape:", output.shape)


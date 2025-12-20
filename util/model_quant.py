# File: util/model_quant.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Module, ModuleList, MaxPool2d
from brevitas.nn import QuantConv2d, QuantIdentity
from brevitas.core.restrict_val import RestrictValueType

# Example quant modules (must exist in your codebase):
from common import CommonActQuant, CommonWeightQuant

DEBUG = False

ENCODER_CHANNELS = [64, 128, 256, 512]
KERNEL_SIZE = 3
POOL_SIZE = 2

class UNetQ(Module):
    """
    A UNet-like encoder network fully quantized with Brevitas, suitable for FINN hardware
    generation. Pads the output to match the input shape for compatibility with mask outputs.
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

        # Final output convolution to match the desired out_channels
        self.out_conv = QuantConv2d(
            in_channels=ENCODER_CHANNELS[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )

        # Optionally initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        # ------------------------------------------------------------------
        # 1) Quantize the input so it becomes INT8 right away
        # ------------------------------------------------------------------
        x = self.in_quant(x)

        # -------------------
        # Forward: Encoder
        # -------------------
        # We have groups of (Conv -> BN -> Act -> Pool)
        for i in range(0, len(self.encoder), 4):
            conv = self.encoder[i]
            bn = self.encoder[i + 1]
            act = self.encoder[i + 2]
            pool = self.encoder[i + 3]

            x = conv(x)
            x = bn(x)
            x = act(x)
            x = pool(x)
            if DEBUG:
                print(f"Encoder block {i // 4}: {x.shape}")

        # --------------
        # Final Output
        # --------------
        x = self.out_conv(x)

        # Zero-pad to match input shape
        output_shape = [x.size(0), x.size(1), x.size(2) * POOL_SIZE ** len(ENCODER_CHANNELS), x.size(3) * POOL_SIZE ** len(ENCODER_CHANNELS)]
        x = F.pad(x, (0, output_shape[3] - x.size(3), 0, output_shape[2] - x.size(2)))
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
    test_input = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        output = model(test_input)

    print("Input shape: ", test_input.shape)
    print("Output shape:", output.shape)

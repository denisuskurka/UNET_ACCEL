import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, MaxPool2d, BatchNorm2d

from brevitas.nn import QuantConv2d, QuantIdentity
from brevitas.core.restrict_val import RestrictValueType

from common import CommonActQuant, CommonWeightQuant  # <--- your custom quant definitions

DEBUG = False

# -------------------------------------------------
#                Encoder
# -------------------------------------------------
class UNetEncoder(Module):
    """
    U-Net Encoder that returns:
      1) The output feature map after the last encoder layer (goes to bottleneck).
      2) A list of skip connections to be used in the decoder.
    """
    def __init__(
        self,
        in_channels,
        encoder_channels=[64, 128, 256, 512],
        kernel_size=3,
        pool_size=2,
        weight_bit_width=8,
        act_bit_width=8
    ):
        super(UNetEncoder, self).__init__()

        self.encoder_blocks = ModuleList()
        prev_channels = in_channels
        for out_ch in encoder_channels:
            block = ModuleList()
            # First convolution
            block.append(
                QuantConv2d(
                    in_channels=prev_channels,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            block.append(BatchNorm2d(out_ch, eps=1e-4))
            block.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width,
                    min_val=0.0,  # ReLU-like
                    max_val=1.0 - 2.0 ** (-act_bit_width),
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO
                )
            )
            # Max pool
            block.append(MaxPool2d(kernel_size=pool_size))

            self.encoder_blocks.append(block)
            prev_channels = out_ch

        self.out_channels = encoder_channels[-1]

    def forward(self, x):
        skip_connections = []
        for block in self.encoder_blocks:
            # block is [Conv, BN, Activation, MaxPool]
            conv = block[0]
            bn = block[1]
            act = block[2]
            pool = block[3]

            x = conv(x)
            x = bn(x)
            x = act(x)
            # Save the feature *before* pooling (skip)
            skip_connections.append(x)
            if DEBUG:
                print(f"Encoder conv out: {x.shape}")
            x = pool(x)
            if DEBUG:
                print(f"Encoder after pool: {x.shape}")

        return x, skip_connections

# -------------------------------------------------
#               Bottleneck
# -------------------------------------------------
class UNetBottleneck(Module):
    """
    U-Net bottleneck block. Just a small stack of convs in the middle.
    """
    def __init__(
        self,
        in_channels,
        bottleneck_channels=1024,
        kernel_size=3,
        weight_bit_width=8,
        act_bit_width=8
    ):
        super(UNetBottleneck, self).__init__()

        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )
        self.bn = BatchNorm2d(bottleneck_channels, eps=1e-4)
        self.act = QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=act_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-act_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO
        )
        self.out_channels = bottleneck_channels

        # You can add more layers if you like (as in a typical “double conv”)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if DEBUG:
            print(f"Bottleneck output: {x.shape}")
        return x

# -------------------------------------------------
#               Decoder
# -------------------------------------------------
class UNetDecoder(Module):
    """
    U-Net decoder that:
      - Upsamples the bottleneck output
      - Concatenates skip connections
      - Applies additional conv layers
    We replace ConvTranspose2d with upsampling + normal QuantConv2d.
    """
    def __init__(
        self,
        decoder_channels=[512, 256, 128, 64],
        skip_channels=[64, 128, 256, 512],  # must match your encoder
        bottleneck_channels=1024,
        kernel_size=3,
        weight_bit_width=8,
        act_bit_width=8,
        out_channels=1  # final output channels
    ):
        super(UNetDecoder, self).__init__()

        # We'll build multiple decoder stages, each upsamples + merges skip
        # The i-th stage has in_channels from previous stage (or bottleneck),
        # skip_channels[i], and out_channels = decoder_channels[i].
        self.decoder_stages = ModuleList()
        in_ch = bottleneck_channels

        for i, out_ch in enumerate(decoder_channels):
            stage = DecoderBlock(
                in_ch,
                skip_ch=skip_channels[-(i+1)],  # skip from the encoder in reverse order
                out_ch=out_ch,
                kernel_size=kernel_size,
                weight_bit_width=weight_bit_width,
                act_bit_width=act_bit_width
            )
            self.decoder_stages.append(stage)
            in_ch = out_ch  # next stage’s input = this stage’s out

        # Final 1x1 conv to produce the required out_channels
        self.out_conv = QuantConv2d(
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )

    def forward(self, bottleneck_x, skip_connections):
        """
        :param bottleneck_x: Output of the bottleneck
        :param skip_connections: List of features from the encoder (before pooling),
                                 in the same order they were generated.
        """
        x = bottleneck_x

        # skip_connections were appended at each encoder block in forward order
        # So the skip for the first decoder block is skip_connections[-1], etc.
        # We'll process them in reverse.

        for i, stage in enumerate(self.decoder_stages):
            skip = skip_connections[-(i+1)]
            x = stage(x, skip)

        x = self.out_conv(x)
        return x

# -------------------------------------------------
#        DecoderBlock (upsample + conv)
# -------------------------------------------------
class DecoderBlock(Module):
    """
    Single decoder block:
      1) Upsample by factor 2 (nearest)
      2) Conv -> BN -> Act
      3) Concat skip
      4) Another Conv -> BN -> Act
    """
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        kernel_size=3,
        weight_bit_width=8,
        act_bit_width=8
    ):
        super(DecoderBlock, self).__init__()

        # 1) after upsampling, we do conv to out_ch
        self.up_conv = QuantConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )
        self.up_bn = BatchNorm2d(out_ch, eps=1e-4)
        self.up_act = QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=act_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-act_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO
        )

        # 2) after concatenation, we do another conv to out_ch
        concat_in_ch = out_ch + skip_ch
        self.conv = QuantConv2d(
            in_channels=concat_in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )
        self.conv_bn = BatchNorm2d(out_ch, eps=1e-4)
        self.conv_act = QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=act_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-act_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO
        )

    def forward(self, x, skip):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_conv(x)
        x = self.up_bn(x)
        x = self.up_act(x)

        # Ensure matching spatial dimensions for skip
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')

        # Concat
        x = torch.cat([x, skip], dim=1)

        # Conv block
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_act(x)
        return x

# -------------------------------------------------
#         Wrapper: UNetSplit
# -------------------------------------------------
class UNetSplit(Module):
    """
    A convenience module that ties together:
      - UNetEncoder
      - UNetBottleneck
      - UNetDecoder
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64],
        bottleneck_channels=1024,
        weight_bit_width=8,
        act_bit_width=8
    ):
        super(UNetSplit, self).__init__()

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width
        )

        self.bottleneck = UNetBottleneck(
            in_channels=encoder_channels[-1],
            bottleneck_channels=bottleneck_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width
        )

        self.decoder = UNetDecoder(
            decoder_channels=decoder_channels,
            skip_channels=encoder_channels,  # must match
            bottleneck_channels=bottleneck_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            out_channels=out_channels
        )

    def forward(self, x):
        # 1) Encoder
        x_encoded, skip_list = self.encoder(x)

        # 2) Bottleneck
        x_bottleneck = self.bottleneck(x_encoded)

        # 3) Decoder
        x_out = self.decoder(x_bottleneck, skip_list)

        # Optionally, you might want to resize to the original input size
        # if needed (as in your original code). We'll do that here:
        original_size = x.size()[2:]
        if x_out.size()[2:] != original_size:
            x_out = F.interpolate(x_out, size=original_size, mode='bilinear', align_corners=False)

        return x_out


# -------------------------------------------------
#     Example usage
# -------------------------------------------------
if __name__ == "__main__":
    # Example configuration using a simple dictionary-like class
    class Config:
        def getint(self, section, option, fallback=None):
            config = {
                'QUANT': {
                    'WEIGHT_BIT_WIDTH': 8,
                    'ACT_BIT_WIDTH': 8,
                    'IN_BIT_WIDTH': 8
                },
                'MODEL': {
                    'NUM_CLASSES': 1,
                    'IN_CHANNELS': 1,
                    'OUT_CHANNELS': 1
                }
            }
            return config.get(section, {}).get(option, fallback)

    cfg = Config()

    # Get config
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS', fallback=1)
    out_channels = cfg.getint('MODEL', 'OUT_CHANNELS', fallback=1)
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH', fallback=8)
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH', fallback=8)
    in_bit_width   = cfg.getint('QUANT', 'IN_BIT_WIDTH', fallback=8)  # Typically used for your first layer quant

    # Instantiate split U-Net
    model = UNetSplit(
        in_channels=in_channels,
        out_channels=out_channels,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width
    )
    model.eval()

    # Testing with a sample input
    test_input = torch.randn(1, in_channels, 160, 160)
    with torch.no_grad():
        output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)

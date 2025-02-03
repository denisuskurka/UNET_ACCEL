import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, MaxPool2d, BatchNorm2d
from brevitas.nn import QuantConv2d, QuantIdentity
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant_tensor import QuantTensor

DEBUG = False

# Suppose you have custom quant definitions in common.py
# from common import CommonActQuant, CommonWeightQuant

class UNetEncoder(Module):
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
            # First conv
            block.append(
                QuantConv2d(
                    in_channels=prev_channels,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    bias=False,
                    # weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            block.append(BatchNorm2d(out_ch, eps=1e-4))
            block.append(
                QuantIdentity(
                    # act_quant=CommonActQuant,
                    bit_width=act_bit_width,
                    min_val=0.0,
                    max_val=1.0 - 2.0 ** (-act_bit_width),
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                    return_quant_tensor=True
                )
            )
            block.append(MaxPool2d(kernel_size=pool_size))
            self.encoder_blocks.append(block)
            prev_channels = out_ch
        self.out_channels = encoder_channels[-1]

    def forward(self, x):
        skip_connections = []
        for block in self.encoder_blocks:
            conv = block[0]
            bn = block[1]
            act = block[2]
            pool = block[3]

            x = conv(x)
            x = bn(x)
            x = act(x)  # x might become a QuantTensor here
            skip_connections.append(x)  # store skip
            x = pool(x.tensor if isinstance(x, QuantTensor) else x)
        return x, skip_connections


class UNetBottleneck(Module):
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
            # weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width
        )
        self.bn = BatchNorm2d(bottleneck_channels, eps=1e-4)
        self.act = QuantIdentity(
            # act_quant=CommonActQuant,
            bit_width=act_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-act_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            return_quant_tensor=True
        )
        self.out_channels = bottleneck_channels

    def forward(self, x):
        # if x is QuantTensor, use x.tensor for conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(Module):
    """
    Single decoder block:
      1) Upsample by factor 2
      2) Conv -> BN -> Act
      3) Concat skip
      4) Another Conv -> BN -> Act
    We must dequantize skip & x before cat if they have different scaling.
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
        self.up_conv = QuantConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=weight_bit_width
        )
        self.up_bn = BatchNorm2d(out_ch, eps=1e-4)
        self.up_act = QuantIdentity(
            bit_width=act_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-act_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            return_quant_tensor=True
        )
        concat_in_ch = out_ch + skip_ch
        self.conv = QuantConv2d(
            in_channels=concat_in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=weight_bit_width
        )
        self.conv_bn = BatchNorm2d(out_ch, eps=1e-4)
        self.conv_act = QuantIdentity(
            bit_width=act_bit_width,
            min_val=0.0,
            max_val=1.0 - 2.0 ** (-act_bit_width),
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            return_quant_tensor=True
        )

    def forward(self, x, skip):
        # Upsample
        if isinstance(x, QuantTensor):
            up_in = x.tensor
        else:
            up_in = x
        up_in = F.interpolate(up_in, scale_factor=2, mode='nearest')
        # First conv
        x = self.up_conv(up_in)
        x = self.up_bn(x)
        x = self.up_act(x)  # might be QuantTensor

        # Dequantize both x and skip before cat
        if isinstance(x, QuantTensor):
            x = x.tensor
        if isinstance(skip, QuantTensor):
            skip = skip.tensor

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')

        x = torch.cat([x, skip], dim=1)  # float Tensors now
        # second conv
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_act(x)
        return x


class UNetDecoder(Module):
    def __init__(
        self,
        decoder_channels=[512, 256, 128, 64],
        skip_channels=[64, 128, 256, 512],
        bottleneck_channels=1024,
        kernel_size=3,
        weight_bit_width=8,
        act_bit_width=8,
        out_channels=1
    ):
        super(UNetDecoder, self).__init__()
        self.decoder_stages = ModuleList()
        in_ch = bottleneck_channels
        for i, out_ch in enumerate(decoder_channels):
            stage = DecoderBlock(
                in_ch,
                skip_ch=skip_channels[-(i+1)],
                out_ch=out_ch,
                kernel_size=kernel_size,
                weight_bit_width=weight_bit_width,
                act_bit_width=act_bit_width
            )
            self.decoder_stages.append(stage)
            in_ch = out_ch
        # final 1x1
        self.out_conv = QuantConv2d(
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_bit_width=weight_bit_width
        )

    def forward(self, bottleneck_x, skip_connections):
        x = bottleneck_x
        for i, stage in enumerate(self.decoder_stages):
            skip = skip_connections[-(i+1)]
            x = stage(x, skip)
        x = self.out_conv(x)
        return x


class UNetSplit(Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64],
        bottleneck_channels=1024,
        weight_bit_width=8,
        act_bit_width=8
    ):
        super(UNetSplit, self).__init__()
        self.encoder = UNetEncoder(
            in_channels=in_ch,
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
            skip_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            out_channels=out_ch
        )

    def forward(self, x):
        x_encoded, skip_list = self.encoder(x)
        x_bottleneck = self.bottleneck(x_encoded)
        x_out = self.decoder(x_bottleneck, skip_list)
        # optional resize
        orig_size = x.shape[2:]
        if x_out.shape[2:] != orig_size:
            x_out = F.interpolate(x_out, size=orig_size, mode='bilinear', align_corners=False)
        return x_out


if __name__ == "__main__":
    # quick test
    model = UNetSplit(in_ch=1, out_ch=1).eval()
    inp = torch.randn(1,1,128,128)
    out = model(inp)
    print("In:", inp.shape, "Out:", out.shape)

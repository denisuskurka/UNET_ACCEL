import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU

from common import CommonActQuant
from common import CommonWeightQuant
from tensor_norm import TensorNorm

DEBUG = False

class UNetSplit(Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64],
        bottleneck_channels=1024,
        weight_bit_width=1,
        act_bit_width=1
    ):

        super(UNetSplit, self).__init__()
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=weight_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        self.conv_features.append(
            QuantConv2d(
                kernel_size=1,
                in_channels=in_ch,
                out_channels=1,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
        in_ch = out_ch
        self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
        self.conv_features.append(
            QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for nmod in self.conv_features:
            x = nmod(x)
        return x

if __name__ == "__main__":
    # quick test
    model = UNetSplit(in_ch=1, out_ch=1).eval()
    inp = torch.randn(1,1,128,128)
    out = model(inp)
    print("In:", inp.shape, "Out:", out.shape)

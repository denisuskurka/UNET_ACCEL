import torch
from torch.nn import BatchNorm2d, MaxPool2d, Module, ModuleList, ConvTranspose2d
from torch.nn import functional as F

from brevitas.nn import QuantConv2d, QuantConvTranspose2d, QuantIdentity

from common import CommonActQuant, CommonWeightQuant  # Importing from common.py
from brevitas.core.restrict_val import RestrictValueType

DEBUG = False

# U-Net Configuration Constants
ENCODER_CHANNELS = [64, 128, 256, 512]
DECODER_CHANNELS = [512, 256, 128, 64]
BOTTLE_NECK_CHANNELS = 1024
KERNEL_SIZE = 3
POOL_SIZE = 2

class UNet(Module):
    def __init__(self, in_channels, out_channels, weight_bit_width=8, act_bit_width=8, in_bit_width=8):
        super(UNet, self).__init__()

        # Encoder Path
        self.encoder = ModuleList()
        prev_channels = in_channels
        for out_ch in ENCODER_CHANNELS:
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
            self.encoder.append(BatchNorm2d(out_ch, eps=1e-4))
            self.encoder.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width,
                    min_val=0.0,  # Assuming ReLU-like activation
                    max_val=1.0 - 2.0 ** (-act_bit_width),
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO
                )
            )
            self.encoder.append(MaxPool2d(kernel_size=POOL_SIZE))
            prev_channels = out_ch

        # Bottleneck
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

        # Decoder Path
        self.decoder = ModuleList()
        current_channels = BOTTLE_NECK_CHANNELS
        for out_ch in DECODER_CHANNELS:
            # Upsample
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
            # Convolution after concatenation
            concatenated_channels = out_ch + (current_channels // 2)  # Concatenated channels
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
            # Adjust BatchNorm2d to match concatenated channels
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
            current_channels = out_ch  # Update for the next iteration

        # Final Output Layer
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

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantConvTranspose2d):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        encoder_outputs = []

        # Save the original input dimensions
        original_size = x.size()[2:]

        # Encoder Path
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, MaxPool2d):
                encoder_outputs.append(x)
                if DEBUG:
                    print(f"Encoder output at layer: {x.shape}")

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x)
        if DEBUG:
            print(f"Bottleneck output: {x.shape}")

        # Decoder Path
        for i in range(0, len(self.decoder), 6):
            # Upsample
            up_conv = self.decoder[i]
            up_bn = self.decoder[i+1]
            up_act = self.decoder[i+2]
            x = up_conv(x)
            x = up_bn(x)
            x = up_act(x)
            if DEBUG:
                print(f"After upsample: {x.shape}")

            # Retrieve corresponding encoder output for skip connection
            enc_output = encoder_outputs[-(i // 6 + 1)]
            if DEBUG:
                print(f"Skip connection: {enc_output.shape}")

            # Ensure matching spatial dimensions
            if x.size() != enc_output.size():
                x = F.interpolate(x, size=enc_output.shape[2:], mode='nearest')

            # Concatenate along channel dimension
            x = torch.cat([x, enc_output], dim=1)
            if DEBUG:
                print(f"After concatenation: {x.shape}")

            # Convolutional Block
            conv = self.decoder[i+3]
            bn = self.decoder[i+4]
            act = self.decoder[i+5]
            x = conv(x)
            x = bn(x)
            x = act(x)
            if DEBUG:
                print(f"After convolution block: {x.shape}")

        # Final Output
        x = self.out_conv(x)

        # Resize to match the original input dimensions
        if x.size()[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

        return x


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
    
    # Extract configuration values for in_channels and out_channels
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS', fallback=1)
    out_channels = cfg.getint('MODEL', 'OUT_CHANNELS', fallback=1)
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH', fallback=8)
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH', fallback=8)
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH', fallback=8)
    
    # Instantiate the UNet model with the correct arguments
    model = UNet(
        in_channels=in_channels, 
        out_channels=out_channels,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width
    )
    model.eval()

    # Example input: (batch_size=1, in_channels=1, height=128, width=128)
    test_input = torch.randn(1, in_channels, 160, 160)
    with torch.no_grad():
        output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)
    # Expected output shape: (1, out_channels, 128, 128)

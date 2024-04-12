import torch
import torchvision
from torch import nn


class DeformableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False
    ):
        super(DeformableConv2d, self).__init__()

        self.offset_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.
        print("Input shape:", x.shape)

        offset = self.offset_conv(x)
        print("Offset shape:", offset.shape)
        offset = offset.clamp(-max_offset, max_offset)

        modulator = self.modulator_conv(x)
        modulator = 2. * torch.sigmoid(modulator)
        print("Modulator shape:", modulator.shape)

        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.conv.weight,
            bias=self.conv.bias,
            padding=self.conv.padding,
            stride=self.conv.stride,
            mask=modulator,
        )
        return x
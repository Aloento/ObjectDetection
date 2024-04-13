from torch import nn, Tensor

from DeformableConv2d import DeformableConv2d


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            dcn: bool = False,
    ):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = DeformableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ) if dcn else nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        down_sample_channels = in_channels // 2

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=down_sample_channels,
            kernel_size=1,
            stride=1,
            dcn=False
        )
        self.conv2 = ConvBlock(
            in_channels=down_sample_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dcn=True
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x.clone()

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity

        return out

from torch import nn

from ResNet import ConvBlock


class YoloHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(YoloHead, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            dcn=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

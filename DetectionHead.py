from torch import nn

from Darknet import ConvBlock


class ExtractBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ExtractBlock, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dcn=False
        )
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=1,
            dcn=False
        )
        self.conv3 = ConvBlock(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dcn=False
        )
        self.conv4 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=1,
            dcn=False
        )
        self.conv5 = ConvBlock(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dcn=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

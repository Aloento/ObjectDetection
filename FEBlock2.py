from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from torch import nn


class FEBlock2(nn.Module):
    """
    1. DSC

       in_channels: 16
       out_channels: 32
       kernel_size: 3
       stride: 1
       padding: 1

    2. BatchNorm2d
    3. PReLU
    4. MaxPool2d

       kernel_size: 2
       stride: 2
    """

    def __init__(self):
        super(FEBlock2, self).__init__()
        self.in_channels = 16
        self.out_channels = 32

        self.depthwise = DepthwiseSeparableConvolution(
            in_ch=self.in_channels,
            out_ch=self.out_channels
        )

        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.lu = nn.PReLU(num_parameters=self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.lu(x)
        x = self.pool(x)
        return x

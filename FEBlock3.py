from fightingcv_attention.attention.CBAM import CBAMBlock
from torch import nn


class FEBlock3(nn.Module):
    """
    1. Dilated Conv2d

       in_channels: 32
       out_channels: 64
       kernel_size: 3
       stride: 1
       padding: 2
       dilation: 2

    2. BatchNorm2d
    3. PReLU
    4. MaxPool2d

       kernel_size: 2
       stride: 2

    5. CBAM
    """

    def __init__(self):
        super(FEBlock3, self).__init__()
        self.in_channels = 32
        self.out_channels = 64

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2
        )

        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.lu = nn.PReLU(num_parameters=self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbam = CBAMBlock(channel=self.out_channels, kernel_size=7)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.lu(x)
        x = self.pool(x)
        x = self.cbam(x)
        return x, residual

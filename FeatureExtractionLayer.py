import torchvision.ops
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from torch import nn
from torchvision.ops import DropBlock2d


class FEBlock1(nn.Module):
    """
    1. Conv2d

       in_channels: 3
       out_channels: 16
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
        super(FEBlock1, self).__init__()
        self.in_channels = 3
        self.out_channels = 16

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.lu = nn.PReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lu(x)
        x = self.pool(x)
        return x


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
        self.lu = nn.PReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.lu(x)
        x = self.pool(x)
        return x


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
        self.lu = nn.PReLU()
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


class FEBlock4(nn.Module):
    """
    1. DSC

       ```python
       in_channels: 64
       out_channels: 128
       kernel_size: 3
       stride: 1
       padding: 1
       ```

    2. output += Conv(ResidualBlock)
    3. BatchNorm2d
    4. PReLU
    5. DropBlock
    """

    def __init__(self):
        super(FEBlock4, self).__init__()
        self.in_channels = 64
        self.out_channels = 128

        self.depthwise = DepthwiseSeparableConvolution(
            in_ch=self.in_channels,
            out_ch=self.out_channels
        )

        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.lu = nn.PReLU()
        self.dropblock = DropBlock2d(p=0.3, block_size=3)

        self.residual = nn.Conv2d(
            in_channels=self.in_channels // 2,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x, residual):
        x = self.depthwise(x)

        residual = self.residual(residual)
        x += residual

        x = self.bn(x)
        x = self.lu(x)
        x = self.dropblock(x)
        return x

from torch import nn


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
       kernel_size: 3
       stride: 1
       padding: 1

    2. Pointwise

       in_channels: 16
       out_channels: 32
       kernel_size: 1
       stride: 1
       padding: 0

    3. BatchNorm2d
    4. PReLU
    5. MaxPool2d

       kernel_size: 2
       stride: 2
    """

    def __init__(self):
        super(FEBlock2, self).__init__()
        self.in_channels = 16
        self.out_channels = 32

        self.depthwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
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

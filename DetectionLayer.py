from torch import nn, Tensor

from EnhancementBlock import EnhancementBlock


class DetectionLayer(nn.Module):
    def __init__(self, in_channels: int):
        super(DetectionLayer, self).__init__()
        self.enhance = EnhancementBlock(in_channels)
        self.out_channels = 22

        self.conv = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.lu = nn.PReLU(self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.enhance(x)
        x = self.conv(x)
        x = self.lu(x)
        x = self.bn(x)
        return x

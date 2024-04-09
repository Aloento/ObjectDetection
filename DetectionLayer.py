from torch import nn, Tensor

from EnhancementBlock import EnhancementBlock


class DetectionLayer(nn.Module):
    """
    1. Conv2d

       in_channels: 128
       out_channels: 22
       kernel_size: 1
       stride: 1
       padding: 0

       [x Center, y Center, Width, Height, Confidence, Class]

    2. Sigmoid
    """
    def __init__(self):
        super(DetectionLayer, self).__init__()
        self.enhance = EnhancementBlock()

        self.in_channels = 128
        self.out_channels = 22

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.enhance(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

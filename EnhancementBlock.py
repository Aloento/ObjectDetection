from fightingcv_attention.attention.CBAM import CBAMBlock
from torch import nn
from torchvision.ops import DeformConv2d


class EnhancementBlock(nn.Module):
    """
    1. DCN v2

       in_channels: 128
       out_channels: 128
       kernel_size: 3
       stride: 1
       padding: 1

    2. BatchNorm2d
    3. PReLU
    4. CBAM
    """

    def __init__(self):
        super(EnhancementBlock, self).__init__()
        self.in_channels = 128
        self.out_channels = 128

        self.dcn = DeformConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn = nn.BatchNorm2d(self.out_channels)
        self.lu = nn.PReLU(num_parameters=self.out_channels)
        self.cbam = CBAMBlock(channel=self.out_channels, kernel_size=7)

    def forward(self, x):
        x = self.dcn(x)
        x = self.bn(x)
        x = self.lu(x)
        x = self.cbam(x)
        return x

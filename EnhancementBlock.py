from fightingcv_attention.attention.CBAM import CBAMBlock
from torch import nn, Tensor
from torchvision.ops import DeformConv2d


class EnhancementBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(EnhancementBlock, self).__init__()

        self.dcn = DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.offset = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * 3 * 3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.lu = nn.PReLU(num_parameters=in_channels)
        self.cbam = CBAMBlock(channel=in_channels, kernel_size=7)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x)
        x = self.dcn(x, offset)
        x = self.bn(x)
        x = self.lu(x)
        x = self.cbam(x)
        return x

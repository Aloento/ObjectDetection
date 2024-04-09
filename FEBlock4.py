from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from torch import nn, Tensor
from torchvision.ops import DropBlock2d


class FEBlock4(nn.Module):
    """
    1. DSC

       in_channels: 64
       out_channels: 128
       kernel_size: 3
       stride: 1
       padding: 1

    2. BatchNorm2d
    3. PReLU
    4. DropBlock
    5. output += Conv(ResidualBlock)
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
        self.lu = nn.PReLU(num_parameters=self.out_channels)
        self.dropblock = DropBlock2d(p=0.3, block_size=3)

        self.residual_conv = nn.Conv2d(
            in_channels=self.in_channels // 2,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1
        )
        self.residual_pool = nn.AdaptiveAvgPool2d(output_size=(80, 80))

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.lu(x)
        x = self.dropblock(x)

        residual = self.residual_conv(residual)
        residual = self.residual_pool(residual)

        assert x.shape == residual.shape, f"Output shape {x.shape} and residual shape {residual.shape} must be equal"
        x += residual
        return x

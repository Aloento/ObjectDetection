from torch import nn, Tensor
from torch.nn import Sequential
from torchvision.ops import DropBlock2d

from ResBlock import ResBlock


def make_res_block(out_channels: int, stride: int) -> Sequential:
    strides = [stride, 1]
    layers = []
    in_channels = 64

    for stride in strides:
        layers.append(ResBlock(in_channels, out_channels, stride))
        in_channels = out_channels

    return nn.Sequential(*layers)


class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        self.input_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(64)
        self.lu = nn.PReLU(num_parameters=64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block1 = make_res_block(64, 1)
        self.res_block2 = make_res_block(128, 2)
        self.res_block3 = make_res_block(256, 2)
        self.res_block4 = make_res_block(512, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropblock = DropBlock2d(block_size=3, p=0.1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = self.lu(out)
        out = self.max_pool(out)

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.dropblock(out)

        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.avg_pool(out)

        return out

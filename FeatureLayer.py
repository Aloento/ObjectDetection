from torch import nn, Tensor
from torch.nn import Sequential
from torchvision.ops import DropBlock2d

from EnhancementBlock import EnhancementBlock
from ResBlock import ResBlock


class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()
        self.in_channels = 64

        self.input_conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(self.in_channels)
        self.lu = nn.PReLU(num_parameters=self.in_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.enhance = EnhancementBlock(self.in_channels)

        self.res_block1 = self.make_res_block(self.in_channels, 1)
        self.res_block2 = self.make_res_block(128, 2)

        self.dropblock = DropBlock2d(block_size=3, p=0.3)

        self.res_block3 = self.make_res_block(256, 2)
        self.res_block4 = self.make_res_block(512, 2)

    def make_res_block(self, out_channels: int, stride: int) -> Sequential:
        strides = [stride, 1]
        layers = []

        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = self.lu(out)
        out = self.max_pool(out)

        out = self.enhance(out)

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.dropblock(out)

        out = self.res_block3(out)
        out = self.res_block4(out)
        return out

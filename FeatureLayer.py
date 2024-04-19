from torch import nn, Tensor
from torch.nn import Sequential

from Bottleneck import Bottleneck


class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()
        self.in_planes = 64

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_res_block(64, 3, 1)
        self.layer2 = self.make_res_block(128, 4, 2)
        self.layer3 = self.make_res_block(256, 23, 2)
        self.layer4 = self.make_res_block(512, 3, 2)

    def make_res_block(self, planes: int, blocks: int, stride: int = 1) -> Sequential:
        down_sample = None
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            down_sample = Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            )

        layers = [Bottleneck(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return Sequential(*layers)

    def forward(self, x: Tensor):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

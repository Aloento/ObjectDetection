import torch
from torch import nn, Tensor
from torch.nn import Sequential

from Bottleneck import Bottleneck

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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
        self.lu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block1 = self.make_res_block(64, 3, 1)
        self.res_block2 = self.make_res_block(128, 4, 2)
        self.res_block3 = self.make_res_block(256, 6, 2)
        self.res_block4 = self.make_res_block(512, 3, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, len(classes))

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

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.lu(out)
        out = self.pool(out)

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

import torch
from torch import nn, Tensor
from torch.nn import Sequential

from Bottleneck import Bottleneck
from VOCDataset import catalogs


class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_res_block(64, 3, 1)
        self.layer2 = self.make_res_block(128, 4, 2)
        self.layer3 = self.make_res_block(256, 6, 2)
        self.layer4 = self.make_res_block(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, len(catalogs))

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

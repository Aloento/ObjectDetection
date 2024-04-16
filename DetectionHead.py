import torch
from torch import nn

from Bottleneck import Bottleneck


class DetectionHead(nn.Module):
    def __init__(self):
        super(DetectionHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=512 * Bottleneck.expansion,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 7 * 7, 4)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    detection_head = DetectionHead()

    x = torch.randn(1, 512 * Bottleneck.expansion, 7, 7)
    h = detection_head(x)
    print(h.shape)

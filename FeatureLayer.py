import torch
from torch import nn, Tensor
from torch.nn import Sequential

from Darknet import ConvBlock
from VOCDataset import catalogs


class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()
        self.input_seq = nn.Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                dcn=False
            ),
            ConvBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dcn=True
            )
        )


    def forward(self, x: Tensor) -> Tensor:

        return out

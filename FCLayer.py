from torch import nn, Tensor


class FCLayer(nn.Module):
    def __init__(self, in_channels: int):
        super(FCLayer, self).__init__()
        self.out_channels = 22

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, self.out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

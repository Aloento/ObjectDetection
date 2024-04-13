from torch import nn, Tensor
from torch.nn import Sequential

from Darknet import ConvBlock, ResBlock


class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()
        self.input_conv = ConvBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dcn=False
        )

        self.x1_seq = Sequential(
            ConvBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dcn=False
            ),
            ResBlock(in_channels=64, out_channels=64)
        )

        self.x2_seq = Sequential(
            ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dcn=False
            ),
            ResBlock(in_channels=128, out_channels=128),
            ResBlock(in_channels=128, out_channels=128)
        )

        self.x8_seq1 = Sequential(
            ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                dcn=False
            ),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),

            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256)
        )

        self.x8_seq2 = Sequential(
            ConvBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                dcn=False
            ),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),

            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512)
        )

        self.x4_seq = Sequential(
            ConvBlock(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                dcn=False
            ),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024)
        )

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor):
        inp = self.input_conv(x)

        x1 = self.x1_seq(inp)
        x2 = self.x2_seq(x1)
        x8_1 = self.x8_seq1(x2)
        x8_2 = self.x8_seq2(x8_1)
        x4 = self.x4_seq(x8_2)

        # x8, x16, x32
        return x8_1, x8_2, x4


if __name__ == "__main__":
    from torchsummary import summary
    import torch

    model = FeatureLayer()
    summary(model, (3, 416, 416), device="cpu")

    x = torch.randn((1, 3, 416, 416))
    y1, y2, y3 = model(x)
    print(y1.shape, y2.shape, y3.shape)

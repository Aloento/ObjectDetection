import numpy as np
from torch import nn, Tensor

from ExtractBlock import ExtractBlock
from FeatureLayer import FeatureLayer
from ResNet import ConvBlock
from VOCDataset import catalogs
from YoloHead import YoloHead


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_classes = len(catalogs)
        self.num_anchors = (self.num_classes + 1 + 4) * 3

        self.darknet = FeatureLayer()

        self.feature1 = ExtractBlock(in_channels=1024, out_channels=512)
        self.head1 = YoloHead(in_channels=512, mid_channels=1024, out_channels=self.num_anchors)

        self.conv1 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, dcn=False)
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        self.feature2 = ExtractBlock(in_channels=768, out_channels=256)
        self.head2 = YoloHead(in_channels=256, mid_channels=512, out_channels=self.num_anchors)

        self.conv2 = ConvBlock(in_channels=256, out_channels=128, kernel_size=1, stride=1, dcn=False)

        self.feature3 = ExtractBlock(in_channels=384, out_channels=128)
        self.head3 = YoloHead(in_channels=128, mid_channels=256, out_channels=self.num_anchors)

        anchors = [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]
        ]
        self.strides = np.array([8, 16, 32])
        self.anchors = torch.from_numpy((np.array(anchors).T / self.strides).T)

    def forward(self, x: Tensor):
        x8, x16, x32 = self.darknet(x)

        # 13x13 x num_anchors
        feature1 = self.feature1(x32)
        deep = self.head1(feature1)

        # 13x13 -> 26x26
        conv1 = self.conv1(feature1)
        up_sample1 = self.up_sample(conv1)
        mix1 = torch.cat([up_sample1, x16], dim=1)

        # 26x26 x num_anchors
        feature2 = self.feature2(mix1)
        medium = self.head2(feature2)

        # 26x26 -> 52x52
        conv2 = self.conv2(feature2)
        up_sample2 = self.up_sample(conv2)
        mix2 = torch.cat([up_sample2, x8], dim=1)

        # 52x52 x num_anchors
        feature3 = self.feature3(mix2)
        shallow = self.head3(feature3)

        return deep, medium, shallow

    def decode(self, layer: Tensor, layer_type: int):
        # [batch_size, num_anchors, height, width] -> [batch_size, height, width, num_anchors]
        layer = layer.permute(0, 2, 3, 1)  # type: Tensor
        batch_size, height, width, num_anchors = layer.shape
        bboxes = layer.view(batch_size, height, width, 3, self.num_classes + 1 + 4)

        # 3 bbox : (dx, dy), (dw, dh), confidence, classes
        raw_dx_dy = bboxes[:, :, :, :, :2]   # offset of center
        raw_dw_dh = bboxes[:, :, :, :, 2:4]  # offset of width and height
        raw_confi = bboxes[:, :, :, :, 4:5]  # confidence
        raw_proba = bboxes[:, :, :, :, 5:]   # probability of classes

        # generate grid 13x13, 26x26, 52x52
        yv, xv = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        yv = yv.unsqueeze(-1)
        xv = xv.unsqueeze(-1)

        xy_grid = torch.cat([xv, yv], dim=-1)
        # [13, 13, 1, 2] -> [1, 13, 13, 1, 2]
        xy_grid = xy_grid.view(1, height, width, 1, 2)  # type: Tensor
        # [1, height, width, num of (xy) per grid, 2 (has x, y)]
        # [1, 13, 13, 1, 2] -> [batch_size, height, width, 3, 2]
        # then every grid has 3 bbox
        xy_grid = xy_grid.repeat(batch_size, 1, 1, 3, 1).float()

        # calculate bbox center
        pred_xy = (torch.sigmoid(raw_dx_dy) + xy_grid) * self.strides[layer_type]
        pred_wh = (torch.exp(raw_dw_dh) * self.anchors[layer_type]) * self.strides[layer_type]
        pred_xy_wh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_confi)
        pred_prob = torch.sigmoid(raw_proba)

        return torch.concat([pred_xy_wh, pred_conf, pred_prob], dim=-1)


if __name__ == "__main__":
    import torch

    t = torch.randn(1, 3, 416, 416)
    model = Model()
    l, m, s = model(t)

    # [1, 75, 13, 13], [1, 75, 26, 26], [1, 75, 52, 52]
    print(l.shape, m.shape, s.shape)

import torch
from torch import nn, Tensor

from VOCDataset import catalogs


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.ignore_threshold = 0.5

        self.input_size = 416
        self.num_classes = len(catalogs)
        self.bbox_attrs = self.num_classes + 1 + 4

        self.balance = [0.4, 1., 4.]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (self.input_size * self.input_size) / (416 ** 2)  # 5
        self.cls_ratio = 1 * self.num_classes / 80  # 0.25

        self.anchors = [
            [10., 13.],
            [16., 30.],
            [33., 23.],
            [30., 61.],
            [62., 45.],
            [59., 119.],
            [116., 90.],
            [156., 198.],
            [373., 326.]
        ]
        self.anchors_mask = [
            [6, 7, 8],
            [3, 4, 5],
            [0, 1, 2]
        ]

    def forward(
            self,
            # [batch_size, num_anchors, height, width]
            deep: Tensor,  # 13x13
            medium: Tensor,  # 26x26
            shallow: Tensor,  # 52x52
            bboxes: list[Tensor],
            labels: Tensor
    ):
        pass

    def compute_loss(self, pred: Tensor, layer_type: int, target: Tensor):
        batch_size, _, height, width = pred.shape

        # 13 -> 32 pix, 26 -> 16 pix, 52 -> 8 pix
        stride = self.input_size / height

        # relative to the feature map
        scaled_anchors = [
            (anchor_width / stride, anchor_height / stride)
            for anchor_width, anchor_height in self.anchors
        ]

        # [batch_size, num_anchors, height, width] ->
        # [batch_size, 3 bbox, height, width, bbox_attrs]
        prediction = pred.view(
            batch_size,
            len(self.anchors_mask[layer_type]),
            self.bbox_attrs,
            height,
            width
        ).premute(0, 1, 3, 4, 2).contiguous()

        # adjustments parameters for the center of the bounding box
        pred_xs = torch.sigmoid(prediction[..., 0])  # [batch_size, 3, height, width]
        pred_ys = torch.sigmoid(prediction[..., 1])  # [batch_size, 3, height, width]

        # adjustments parameters for the width and height of the bounding box
        pred_ws = prediction[..., 2]  # [batch_size, 3, height, width]
        pred_hs = prediction[..., 3]  # [batch_size, 3, height, width]

        # confidence
        pred_confs = torch.sigmoid(prediction[..., 4])  # [batch_size, 3, height, width]

        # probability of classes
        pred_probs = torch.sigmoid(prediction[..., 5:])  # [batch_size, 3, height, width, num_classes]



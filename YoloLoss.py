import torch
from torch import nn, Tensor

from BBoxOverlaps import calculate_iou
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

    def compute_loss(self, pred: Tensor, layer_type: int, target: list[Tensor]):
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

        ground_truth, no_obj_mask, small_obj_loss_scale = self.transform_target(
            target, layer_type, scaled_anchors, height, width
        )

    def transform_target(
            self,
            target: list[Tensor],
            layer_type: int,
            scaled_anchors: list[tuple[float, float]],
            height: int,
            width: int
    ):
        batch_size = len(target)

        no_obj_mask = torch.ones(
            batch_size, len(self.anchors_mask[layer_type]), height, width
        )

        small_obj_loss_scale = torch.zeros(
            batch_size, len(self.anchors_mask[layer_type]), height, width
        )

        ground_truth = torch.zeros(
            batch_size, len(self.anchors_mask[layer_type]), height, width, self.bbox_attrs
        )

        anchors = torch.tensor(scaled_anchors)
        anchor_boxes = torch.cat([torch.zeros_like(anchors), anchors], dim=1)

        for batch, single_target in enumerate(target):
            if single_target.numel() == 0:
                continue

            # Calculate scaled positions and sizes
            bbox_scaled = torch.zeros_like(single_target)
            bbox_scaled[:, [0, 2]] = single_target[:, [0, 2]] * width
            bbox_scaled[:, [1, 3]] = single_target[:, [1, 3]] * height
            bbox_scaled[:, 4] = single_target[:, 4]

            # Calculate iou and get best anchors
            truth_boxes = torch.cat([torch.zeros_like(bbox_scaled[:, :2]), bbox_scaled[:, 2:4]], dim=1)
            best_anchors = torch.argmax(calculate_iou(truth_boxes, anchor_boxes), dim=-1)

            for i, best_anchor in enumerate(best_anchors):  # type: int, Tensor
                if best_anchor not in self.anchors_mask[layer_type]:
                    continue

                anchor_idx = self.anchors_mask[layer_type].index(best_anchor)
                grid_y = int(bbox_scaled[i, 1])
                grid_x = int(bbox_scaled[i, 0])

                no_obj_mask[batch, anchor_idx, grid_y, grid_x] = 0

                ground_truth[batch, anchor_idx, grid_y, grid_x, :4] = torch.tensor([
                    bbox_scaled[i, 0] - grid_x,  # dx
                    bbox_scaled[i, 1] - grid_y,  # dy
                    torch.log(bbox_scaled[i, 2] / anchors[best_anchor, 0]),  # tw
                    torch.log(bbox_scaled[i, 3] / anchors[best_anchor, 1])  # th
                ])

                ground_truth[batch, anchor_idx, grid_y, grid_x, 4] = 1  # objectness
                ground_truth[batch, anchor_idx, grid_y, grid_x, 5 + int(bbox_scaled[i, 4])] = 1  # class label

                small_obj_loss_scale[batch, anchor_idx, grid_y, grid_x] = (
                        bbox_scaled[i, 2] * bbox_scaled[i, 3] / (height * width)
                )

        return ground_truth, no_obj_mask, small_obj_loss_scale

    def get_ignore_mask(
            self,
            pred_xs: Tensor,
            pred_ys: Tensor,
            pred_ws: Tensor,
            pred_hs: Tensor,
            target: list[Tensor],
            scaled_anchors: list[tuple[float, float]],
            height: int,
            width: int,
            no_obj_mask: Tensor
    ):
        batch_size = len(target)
        
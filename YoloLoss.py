from torch import nn, Tensor


class YoloLoss(nn.Module):
    def __init__(self, strides, anchors, iou_threshold):
        super(YoloLoss, self).__init__()
        self.strides = strides
        self.anchors = anchors
        self.iou_threshold = iou_threshold

    def forward(
            self,
            # [batch_size, num_anchors, height, width]
            deep: Tensor,     # 13x13
            medium: Tensor,   # 26x26
            shallow: Tensor,  # 52x52
            bboxes: list[Tensor],
            labels: Tensor
    ):
        pass

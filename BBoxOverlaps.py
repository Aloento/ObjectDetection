import torch
from torch import Tensor
from torchvision.ops import box_iou


def calculate_iou(bbox_a: Tensor, bbox_b: Tensor) -> Tensor:
    # 转换 box_a 和 box_b 为左上角 (xmin, ymin) 右下角 (xmax, ymax) 的形式
    bbox_a = torch.cat(
        (
            bbox_a[:, :2] - bbox_a[:, 2:] / 2,  # xmin, ymin
            bbox_a[:, :2] + bbox_a[:, 2:] / 2  # xmax, ymax
        ),
        1
    )
    bbox_b = torch.cat(
        (
            bbox_b[:, :2] - bbox_b[:, 2:] / 2,
            bbox_b[:, :2] + bbox_b[:, 2:] / 2
        ),
        1
    )

    iou = box_iou(bbox_a, bbox_b)
    return iou

import torch
from torch import Tensor


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

    # 计算交叉区域的边界坐标
    max_xy = torch.min(bbox_a[:, None, 2:], bbox_b[:, 2:])
    min_xy = torch.max(bbox_a[:, None, :2], bbox_b[:, :2])
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter_area = inter[..., 0] * inter[..., 1]

    # 计算每个框的面积
    area_a = ((bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1]))
    area_b = ((bbox_b[:, 2] - bbox_b[:, 0]) * (bbox_b[:, 3] - bbox_b[:, 1]))

    # 计算并集面积
    union_area = area_a[:, None] + area_b - inter_area

    # 计算 IoU
    iou = inter_area / union_area
    return iou

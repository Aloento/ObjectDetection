import torch
from torch import nn, Tensor
from torchvision.ops import sigmoid_focal_loss

from VOCDataset import catalogs


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = sigmoid_focal_loss

    def forward(self, predictions: Tensor, bboxes: list[Tensor]):
        target_cls = [bbox[:, 4].unique() for bbox in bboxes]

        label_matrix = [
            torch.tensor([1 if i in sublist else 0 for i in range(len(catalogs))])
            for sublist in target_cls
        ]

        label_matrix = torch.stack(label_matrix)
        loss_cls = self.focal(predictions, label_matrix, reduction="mean")

        return loss_cls

import torch
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss

from VOCDataset import catalogs


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCEWithLogitsLoss()

    def forward(self, predictions: Tensor, bboxes: list[Tensor], labels: Tensor):
        target_cls_hot = [
            torch.zeros(len(catalogs)).scatter_(0, label, 1)
            for label in labels
        ]

        label_matrix = torch.stack(target_cls_hot).to(self.device)
        loss_cls = self.bce(predictions, label_matrix)

        return loss_cls

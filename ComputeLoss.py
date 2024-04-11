import torch
from torch import nn, Tensor
from torch.cuda import is_available
from torchvision.ops import sigmoid_focal_loss

from VOCDataset import catalogs


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = sigmoid_focal_loss
        self.device = "cuda" if is_available() else "cpu"

    def forward(self, predictions: Tensor, bboxes: list[Tensor]):
        target_cls = [bbox[:, 4].unique() for bbox in bboxes]

        label_matrix = [
            torch.tensor([1 if i in sublist else 0 for i in range(len(catalogs))], dtype=torch.float32)
            for sublist in target_cls
        ]

        label_matrix = torch.stack(label_matrix).to(self.device)
        loss_cls = self.focal(predictions, label_matrix, reduction="mean")

        return loss_cls

from torch import nn, Tensor
from torchvision.ops import sigmoid_focal_loss


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = sigmoid_focal_loss

    def forward(self, predictions: Tensor, targets: Tensor):
        target_box = targets[:, :4]
        target_cls = targets[:, 5:]

        loss_cls = self.focal(predictions, target_cls, reduction="mean")

        pred = {
            "boxes": target_box,
            "labels": predictions.argmax(dim=1),
            "scores": predictions.max(dim=1).values
        }

        targ = {
            "boxes": target_box,
            "labels": targets[:, 4],
            "scores": target_cls.max(dim=1).values
        }

        return loss_cls, (pred, targ)

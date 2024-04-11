from torch import nn, Tensor
from torchvision.ops import sigmoid_focal_loss


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = sigmoid_focal_loss

    def forward(self, predictions: Tensor, bboxes: Tensor):
        target_boxes = bboxes[:, :4]

        loss_cls = self.focal(predictions, targets, reduction="mean")

        pred = {
            "boxes": target_boxes,
            "labels": predictions.argmax(dim=1),
            "scores": predictions.max(dim=1).values
        }

        targ = {
            "boxes": target_boxes,
            "labels": bboxes[:, 4],
            "scores": target_cls.max(dim=1).values
        }

        return loss_cls, (pred, targ)

from torch import nn, ones_like, Tensor
from torch.nn.functional import one_hot
from torchvision.ops import sigmoid_focal_loss


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = sigmoid_focal_loss
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions: Tensor, targets: Tensor):
        pred_box = predictions[:, :4, 0, 0].squeeze(-1).squeeze(-1)
        pred_box = self.sigmoid(pred_box)
        target_box = targets[:, :4]
        loss_box = self.l1(pred_box, target_box)

        pred_conf = predictions[:, 4, 0, 0].squeeze(-1).squeeze(-1)
        target_conf = ones_like(pred_conf)
        loss_conf = self.bce(pred_conf, target_conf)

        pred_cls = predictions[:, 5:, 0, 0].squeeze(-1).squeeze(-1)
        target_cls = targets[:, 4].long()
        hot_target_cls = one_hot(target_cls, num_classes=17).float()
        loss_cls = self.focal(pred_cls, hot_target_cls, reduction="mean")

        pred = {
            "boxes": pred_box,
            "labels": pred_cls.argmax(dim=1),
            "scores": pred_conf
        }

        targ = {
            "boxes": target_box,
            "labels": target_cls,
            "scores": target_conf
        }

        return (loss_box, loss_conf, loss_cls), (pred, targ)

from torch import nn, ones_like, Tensor
from torch.nn.functional import one_hot
from torchvision.ops import sigmoid_focal_loss


class ComputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.focal = sigmoid_focal_loss

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pred_boxes = predictions[:, :4, :, :]
        target_boxes = targets[:, :4].unsqueeze(-1).unsqueeze(-1).expand_as(pred_boxes)
        loss_boxes = self.smooth_l1(pred_boxes, target_boxes)

        pred_conf = predictions[:, 4, :, :].unsqueeze(1)
        target_conf = ones_like(pred_conf)
        loss_conf = self.focal(pred_conf, target_conf, reduction="sum")

        pred_cls = predictions[:, 5:, :, :]
        target_cls = targets[:, 4].long()
        target_cls = one_hot(target_cls, num_classes=pred_cls.shape[1]).float()
        target_cls = target_cls.unsqueeze(-1).unsqueeze(-1).expand_as(pred_cls)
        loss_cls = self.focal(pred_cls, target_cls, reduction="sum")

        batch_size = predictions.shape[0]
        total_loss = (loss_boxes + loss_conf + loss_cls) / batch_size

        return total_loss

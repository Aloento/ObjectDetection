import torch
from albumentations.core.bbox_utils import convert_bbox_from_albumentations
from torch import nn
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.detection import MeanAveragePrecision


class DetectionMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 17

        self.map = MeanAveragePrecision(box_format="xywh")
        self.precision = Precision(num_classes=self.num_classes, task="multiclass")
        self.recall = Recall(num_classes=self.num_classes, task="multiclass")
        self.f1 = F1Score(num_classes=self.num_classes, task="multiclass")

    def forward(self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        pred_boxes = predictions["boxes"]
        pred_labels = predictions["labels"]
        pred_scores = predictions["scores"]

        targ_boxes = targets["boxes"]
        targ_labels = targets["labels"]
        targ_scores = targets["scores"]

        preds = [{
            "boxes": torch.tensor(
                convert_bbox_from_albumentations(
                    bbox=pred_boxes[i],
                    target_format='coco',
                    rows=640,
                    cols=640
                )
            ).unsqueeze(0).long(),
            "labels": pred_labels[i].unsqueeze(0).long(),
            "scores": pred_scores[i].unsqueeze(0)
        } for i in range(len(pred_boxes))]

        targs = [{
            "boxes": torch.tensor(
                convert_bbox_from_albumentations(
                    bbox=targ_boxes[i],
                    target_format='coco',
                    rows=640,
                    cols=640
                )
            ).unsqueeze(0).long(),
            "labels": targ_labels[i].unsqueeze(0).long(),
            "scores": targ_scores[i].unsqueeze(0)
        } for i in range(len(targ_boxes))]

        map_score = self.map(preds, targs)
        precision_score = self.precision(pred_labels, targ_labels)
        recall_score = self.recall(pred_labels, targ_labels)
        f1_score = self.f1(pred_labels, targ_labels)

        return map_score, precision_score, recall_score, f1_score

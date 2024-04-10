import torch
from albumentations.core.bbox_utils import convert_bbox_from_albumentations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DetectionMetric import DetectionMetric
from Model import Model


def evaluate_epoch(
        model: Model,
        dataloader: DataLoader,
        device: torch.device,
        metric: DetectionMetric,
        writer: SummaryWriter,
        epoch: int
):
    model.eval()
    loop = tqdm(dataloader, leave=True, position=3, desc="Evaluation")

    with torch.no_grad():
        for i, (images, bboxes) in enumerate(loop):  # type: int, (torch.Tensor, torch.Tensor)
            images = images.to(device)
            bboxes = bboxes.to(device)

            _, pred, targ = model(images, bboxes)
            map_score, precision_score, recall_score, f1_score = metric(pred, targ)

            loop.set_postfix(
                map=map_score['map'].item(),
                precision=precision_score.item(),
                recall=recall_score.item(),
                f1=f1_score.item()
            )

            current = epoch * len(dataloader) + i

            writer.add_scalar('Metrics/mAP', map_score['map'], current)
            writer.add_scalar("Metrics/Precision", precision_score, current)
            writer.add_scalar("Metrics/Recall", recall_score, current)
            writer.add_scalar("Metrics/F1", f1_score, current)

            if i % 10 == 0:
                image = images[0]

                pred_box = pred["boxes"][0]
                pred_box_xyxy = convert_bbox_from_albumentations(
                    bbox=pred_box,
                    target_format='pascal_voc',
                    rows=640,
                    cols=640
                )

                pred_label = pred["labels"][0].item()
                pred_score = pred["scores"][0].item()

                target_box = targ["boxes"][0]
                target_box_xyxy = convert_bbox_from_albumentations(
                    bbox=target_box,
                    target_format='pascal_voc',
                    rows=640,
                    cols=640
                )

                box_xyxy = torch.tensor([pred_box_xyxy, target_box_xyxy])
                target_label = targ["labels"][0].item()

                writer.add_image_with_boxes(
                    tag=f'Prediction {current}',
                    img_tensor=image,
                    box_tensor=box_xyxy,
                    labels=[f'Pred: {pred_label} / {pred_score}', f'Target: {target_label}'],
                    global_step=epoch * len(dataloader) + i
                )

    return
